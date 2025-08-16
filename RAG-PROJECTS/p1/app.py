import os
import sys
import hashlib
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PERSIST_DIR = "chroma_db"
DATA_PATH = "data/sample.txt"
HASH_FILE = os.path.join(PERSIST_DIR, ".doc_hash")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

VECTOR_K = 8
VECTOR_FETCH_K = 40
VECTOR_LAMBDA = 0.6

BM25_K = 8

COMPRESS_TOP_K = 6
COMPRESS_SIM_THRESH = 0.2

# --------------------------
# Prompts
# --------------------------
QA_PROMPT_TEMPLATE = """
You are a Q&A assistant that must answer ONLY using the provided context.
If the requested name/place/detail appears anywhere in the context, extract it and answer succinctly (1–2 sentences).
If it does NOT appear in the context, reply exactly:
"I cannot find that in the context."

Do NOT repeat the context or the question.

Context:
{context}

Question:
{question}

Answer:
""".strip()

# Force exactly 4 rewrites for MultiQuery
MQ_PROMPT_TEMPLATE = (
    "Generate 4 diverse, short search queries that could retrieve passages to answer the user question.\n"
    "Question: {question}\n"
    "Queries (one per line):"
)

# --------------------------
# Utilities
# --------------------------
def ensure_file(path: str):
    if not os.path.exists(path):
        print(f"[ERR] File not found: {path}")
        sys.exit(1)

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_saved_hash() -> str | None:
    try:
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def write_hash(value: str):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(value)

def load_documents(path: str) -> List[Document]:
    loader = TextLoader(path, encoding="utf-8")
    docs: List[Document] = loader.load()
    return docs

def load_documents_dir(dir_path: str) -> List[Document]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"{dir_path} is not a directory")
    all_docs: List[Document] = []
    for name in sorted(os.listdir(dir_path)):
        if name.lower().endswith(".txt"):
            p = os.path.join(dir_path, name)
            loader = TextLoader(p, encoding="utf-8")
            all_docs.extend(loader.load())
    return all_docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )
    return splitter.split_documents(docs)

def build_embeddings():
    # Requires: pip install -U langchain-huggingface
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def load_or_build_vectordb(splits, embeddings, force_rebuild: bool = False):
    """
    Reuse existing Chroma index if the DATA_PATH hash matches and not forcing rebuild.
    Otherwise, delete and rebuild the index.
    """
    current_hash = md5_file(DATA_PATH)
    saved_hash = read_saved_hash()

    if (not force_rebuild) and os.path.isdir(PERSIST_DIR) and saved_hash == current_hash:
        # Reuse existing index
        print("[INFO] Reusing existing Chroma index.")
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        return vectordb

    # Rebuild: clear dir, then create fresh
    if os.path.isdir(PERSIST_DIR):
        import shutil
        print("[INFO] Rebuilding Chroma index (content changed or forced)...")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    write_hash(current_hash)
    print("[INFO] Index built & persisted.")
    return vectordb

def build_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_threads=8,       # tune for your CPU
        n_gpu_layers=35,   # tune for your GPU/Metal
        temperature=0.2,
        verbose=False,
    )

def build_hybrid_retriever(splits, vectordb, embeddings, llm):
    # BM25 for exact tokens (e.g., “Baker Street”)
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = BM25_K

    # Vector retriever with MMR
    vector_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": VECTOR_K, "fetch_k": VECTOR_FETCH_K, "lambda_mult": VECTOR_LAMBDA},
    )

    # Blend
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6],
    )

    # Multi-query via custom prompt (forces 4 rewrites)
    mq_prompt = PromptTemplate.from_template(MQ_PROMPT_TEMPLATE)
    question_chain = LLMChain(llm=llm, prompt=mq_prompt)
    multi = MultiQueryRetriever.from_llm(
    retriever=hybrid,
    llm=llm,
    include_original=True,
)

    # Contextual compression
    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        k=COMPRESS_TOP_K,
        similarity_threshold=COMPRESS_SIM_THRESH,
    )
    compressed = ContextualCompressionRetriever(
        base_retriever=multi,
        base_compressor=compressor,
    )
    return compressed

def build_chain(retriever, llm):
    prompt = PromptTemplate(input_variables=["context", "question"], template=QA_PROMPT_TEMPLATE)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )
    return qa

def print_sources(source_docs: List):
    if not source_docs:
        return
    print("\n[SOURCES]")
    seen = set()
    for i, d in enumerate(source_docs, 1):
        src = d.metadata.get("source", "unknown")
        key = (src, d.metadata.get("page", None))
        if key in seen:
            continue
        seen.add(key)
        page = d.metadata.get("page", None)
        page_str = f" (page {page})" if page is not None else ""
        preview = d.page_content.strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "…"
        print(f" {i}. {src}{page_str} :: {preview}")

# Pipeline builder

def build_pipeline(force_rebuild: bool = False):
    ensure_file(DATA_PATH)
    print("[INFO] Loading documents...")
    docs = load_documents(DATA_PATH)  # single file
    # docs = load_documents_dir("data")  # all .txt files
    print("[INFO] Splitting into chunks...")
    splits = split_docs(docs)
    print(f"[INFO] Chunks: {len(splits)}")

    print("[INFO] Loading embeddings...")
    embeddings = build_embeddings()

    print("[INFO] Loading/Building vector store (Chroma)...")
    vectordb = load_or_build_vectordb(splits, embeddings, force_rebuild=force_rebuild)

    print("[INFO] Spinning up LlamaCpp...")
    llm = build_llm()

    print("[INFO] Building hybrid retriever...")
    retriever = build_hybrid_retriever(splits, vectordb, embeddings, llm)

    print("[INFO] Building QA chain...")
    qa = build_chain(retriever, llm)

    return qa

def main():
    print("=== RAG Q&A (context-only answers) ===")
    print("Type your question. Commands: 'rebuild' to reindex, 'exit' to quit.")
    qa = build_pipeline(force_rebuild=False)

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue

        if q.lower() in ("exit", "quit", ":q", ":wq"):
            print("Bye!")
            break

        if q.lower() == "rebuild":
            print("[INFO] Rebuilding index & retriever…")
            qa = build_pipeline(force_rebuild=True)
            continue

        try:
            # LangChain 0.2+ prefers .invoke over __call__
            result = qa.invoke({"query": q})
            answer = result.get("result", "").strip()
            print(f"Bot: {answer}")
            print_sources(result.get("source_documents", []))
        except Exception as e:
            print(f"[ERR] {e}")


if __name__=="__main__":
  main()