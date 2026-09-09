from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def build_vector_store(chunks: list[dict], embedding_model):
    """
    Convert research-paper chunks into searchable documents
    and store their embeddings in a FAISS vector database.
    """

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
            },
        )
        for chunk in chunks
    ]

    vector_store = FAISS.from_documents(
        documents,
        embedding_model,
    )

    return vector_store


def retrieve_relevant_chunks(
    vector_store,
    question: str,
    k: int = 5,
):
    """
    Retrieve the most relevant chunks for a user's question.
    """

    return vector_store.similarity_search(
        question,
        k=k,
    )