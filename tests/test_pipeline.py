from src.chunker import create_chunks
from src.evaluation import (
    calculate_retrieval_relevance,
    calculate_citation_coverage,
    calculate_groundedness,
)


def test_chunk_creation():
    pages = [
        {
            "page": 1,
            "text": "This is a sample research paper section. " * 20,
            "source": "test.pdf",
        }
    ]

    chunks = create_chunks(pages)

    assert len(chunks) > 0
    assert chunks[0]["source"] == "test.pdf"
    assert chunks[0]["page"] == 1
    assert "text" in chunks[0]


def test_retrieval_relevance():
    class Document:
        page_content = "Transformer architecture uses attention mechanisms."
        metadata = {"source": "paper.pdf", "page": 1}

    documents = [Document()]

    score = calculate_retrieval_relevance(
        documents,
        "What attention mechanisms are used?",
    )

    assert score > 0


def test_citation_coverage():
    class Document:
        page_content = "The model uses attention."
        metadata = {"source": "paper.pdf", "page": 3}

    documents = [Document()]

    answer = "The model uses attention [paper.pdf, Page 3]."

    score = calculate_citation_coverage(
        answer,
        documents,
    )

    assert score == 1.0


def test_groundedness():
    class Document:
        page_content = "The transformer uses attention mechanisms."

    documents = [Document()]

    answer = "The transformer uses attention mechanisms."

    score = calculate_groundedness(
        answer,
        documents,
    )

    assert score > 0