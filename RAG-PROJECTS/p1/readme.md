# RAG-Project: Local Document Q\&A Bot

## Overview

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system that allows you to query your own documents with a local Large Language Model (LLM).

The system uses:

* **Mistral 7B Instruct** (GGUF format, running locally with Llama.cpp)
* **Sentence-Transformers (all-mpnet-base-v2)** for generating embeddings
* **ChromaDB** as a vector database to store and retrieve document chunks
* **Hybrid retrieval** combining dense vector search and sparse keyword search (BM25)
* **Streamlit** for a simple interactive web interface

The goal is to help learners understand how a complete RAG pipeline works end-to-end, while running entirely on a local machine.

---

## Why Retrieval-Augmented Generation (RAG)?

Large language models are powerful, but they do not automatically know about your private documents. RAG solves this by:

1. Splitting your documents into **chunks**
2. Converting chunks into **embeddings (vectors)**
3. Storing embeddings in a **vector database**
4. Retrieving the most relevant chunks at query time
5. Supplying those chunks as **context** to the LLM

This ensures the model answers *based on your documents*, not just its pretraining knowledge.

---

## Model Used

We use **Mistral-7B-Instruct** in GGUF format for local inference.
The model can be downloaded from Hugging Face:
[Mistral 7B Instruct GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

The project is flexible and can run with other GGUF models (e.g., Gemma or LLaMA) by updating the `MODEL_PATH`.

---

## Embeddings

We use **sentence-transformers/all-mpnet-base-v2** from Hugging Face to generate dense vector representations of text.
This model balances **accuracy** and **speed**, making it suitable for semantic search tasks.

---

## Libraries and Why We Used Them

* **langchain**: Framework for chaining together LLMs, retrievers, and prompts.
* **langchain-community**: Provides loaders, embeddings, vector stores, and LlamaCpp integration.
* **langchain-huggingface**: For HuggingFace embeddings integration.
* **chromadb**: Lightweight, fast vector database for local storage of embeddings.
* **llama-cpp-python**: Runs GGUF models locally on CPU/GPU (Metal acceleration supported on macOS).
* **sentence-transformers**: Provides pre-trained embedding models such as `all-mpnet-base-v2`.
* **streamlit**: Builds the web interface for asking questions and displaying answers.

---

## Project Structure

```
RAG-PROJECT/
│── data/                # Folder for text files (input documents)
│── models/              # Folder for GGUF models (Mistral, etc.)
│── chroma_db/           # Persistent Chroma database
│── app.py               # Core CLI implementation
│── gui.py               # Streamlit-based web interface
│── README.md            # Documentation
```

---

## Setup Instructions

### 1. Install Conda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don’t already have it.

### 2. Create Environment

```bash
conda create -n ragbot python=3.10 -y
conda activate ragbot
```

### 3. Install Dependencies

```bash
pip install langchain langchain-community langchain-huggingface chromadb llama-cpp-python sentence-transformers streamlit
```

### 4. Download Model

Download Mistral GGUF and place it in `models/`:

```bash
mkdir -p models
# Place Mistral-7B-Instruct-v0.1.Q4_K_M.gguf here
```

### 5. Add Documents

Place your `.txt` files inside the `data/` folder.
Example: `data/sample.txt`

---

## Running the Project

### Start the Web UI

```bash
streamlit run gui.py
```

This launches a local web interface where you can type questions and receive answers based on your documents.

---

## How It Works (Process)

1. **Load Documents**: The project loads `.txt` files from the `data/` folder.
2. **Split into Chunks**: Documents are broken into \~900-character chunks with overlap for context continuity.
3. **Generate Embeddings**: Each chunk is embedded using `all-mpnet-base-v2`.
4. **Store in ChromaDB**: Embeddings are stored in a persistent local vector database (`chroma_db/`).
5. **Hybrid Retrieval**:

   * **BM25** for keyword-based matching
   * **Vector search** for semantic similarity
   * Combined via an **ensemble retriever**
6. **Contextual Compression**: Filters retrieved chunks to the most relevant ones.
7. **Pass to LLM**: The selected context is passed into Mistral, which generates an answer.
8. **Streamlit Display**: The answer and context sources are shown in the browser.

---

## Notes

* The system answers **only based on your documents**. If the answer is not in the context, it responds with *"I cannot find that in the context."*
* Streamlit UI is designed for simplicity; no document uploads are included — documents must be placed in `data/`.
* The retriever is designed to reduce hallucinations by filtering results aggressively.

