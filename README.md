# 🧠 Research Intelligence Platform

An AI-powered research assistant that uses Retrieval-Augmented Generation (RAG) to analyze and compare multiple research papers with evidence-grounded answers and page-level citations.

## 🚀 Overview

The Research Intelligence Platform helps researchers quickly understand academic papers by combining semantic search, local embeddings, vector retrieval, and a locally hosted Large Language Model.

Users can upload multiple PDF research papers and:

- Ask questions about the papers
- Retrieve relevant evidence using semantic search
- Generate answers using a local Llama 3.2 model
- View page-level source citations
- Compare multiple research papers
- Identify research limitations and potential gaps
- Evaluate RAG responses using lightweight heuristic metrics

The system is designed to ground AI-generated answers in retrieved research-paper evidence.

## ✨ Key Features

### 📄 Multi-PDF Processing
Upload and process multiple research papers in PDF format.

### 🔎 Semantic Retrieval
Research-paper text is split into chunks and converted into vector embeddings using Sentence Transformers.

FAISS is used for similarity-based retrieval.

### 🤖 Evidence-Grounded Question Answering
Questions are answered using retrieved research evidence and a local Llama 3.2 model.

### 📚 Page-Level Citations
Retrieved evidence retains the original paper name and page number, allowing users to trace important claims back to their source.

### 📊 Multi-Paper Comparison
Compare papers based on:

- Research objectives
- Methodologies
- Datasets / experimental setup
- Key findings
- Reported limitations
- Similarities and differences

### 🧩 Research Gap Analysis
Analyze papers for:

- Paper-specific limitations
- Unresolved problems
- Common research gaps
- Methodological gaps
- Potential future research directions

### 📈 RAG Evaluation
The system provides lightweight heuristic indicators for:

- Retrieval relevance
- Citation coverage
- Groundedness

These metrics are diagnostic indicators and are not presented as formal benchmark accuracy scores.

## 🏗️ System Architecture

```text
Research Papers (PDF)
        │
        ▼
PDF Text Extraction
        │
        ▼
Text Chunking + Metadata
        │
        ▼
Sentence Transformer Embeddings
        │
        ▼
FAISS Vector Store
        │
        │
   User Question
        │
        ▼
Semantic Retrieval
        │
        ▼
Relevant Evidence
        │
        ▼
Llama 3.2 via Ollama
        │
        ▼
Evidence-Grounded Answer
        │
        ├── Page-Level Citations
        │
        ▼
RAG Evaluation
        │
        ├── Retrieval Relevance
        ├── Citation Coverage
        └── Groundedness
