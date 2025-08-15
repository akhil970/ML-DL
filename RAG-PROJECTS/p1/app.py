import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PERSIST_DIR = "chroma_db"
HASH_FILE = os.path.join(PERSIST_DIR, ".doc_hash")

# --------------------------
# Function to hash files in data/
# --------------------------
def hash_files_in_dir(directory):
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

# --------------------------
# 1. Load documents from PDFs or TXT
# --------------------------
docs = []
for file in os.listdir("data"):
    path = os.path.join("data", file)
    if file.endswith(".txt"):
        loader = TextLoader(path)
        docs.extend(loader.load())
    '''elif file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())'''

# --------------------------
# 2. Split documents into chunks
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# --------------------------
# 3. Create local embeddings
# --------------------------
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --------------------------
# 4. Build or load Chroma DB dynamically
# --------------------------
doc_hash = hash_files_in_dir("data")
rebuild_db = True

if os.path.exists(PERSIST_DIR) and os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        saved_hash = f.read().strip()
    if saved_hash == doc_hash:
        rebuild_db = False

if rebuild_db:
    print("♻️ Documents changed — rebuilding Chroma DB...")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=PERSIST_DIR)
    db.persist()
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(doc_hash)
else:
    print("✅ Using existing Chroma DB...")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)

retriever = db.as_retriever()

# --------------------------
# 5. Load Local Mistral LLM
# --------------------------
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    temperature=0.1,
    max_tokens=512,
    n_threads=4,
    n_gpu_layers=1,
    verbose=False
)

# --------------------------
# 6. Custom Prompt
# --------------------------
prompt_template = """
You are a Question and Answer chat bot answering based on the context provided only 
Don't make the user uncomfortable by wrong answers, just answer according to the question asked in a short summary first.

Your prompt will be in format
Context:{context}
question: {question}

and you should give answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --------------------------
# 7. Create RetrievalQA chain
# --------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Debug: See what retrieval finds for a test question
'''results = retriever.invoke("Who is credited as the author?")
for r in results:
    print(r.page_content[:300])'''

# --------------------------
# 8. Interactive Q&A loop
# --------------------------
print("\n✅ Local Document Q&A Bot (Mistral + Chroma + MiniLM Embeddings)")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa.invoke({"query": query})
    print("Bot:", result["result"])
    #print("\n[DEBUG] Context used:\n", result["source_documents"])
