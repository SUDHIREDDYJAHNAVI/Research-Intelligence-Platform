def calculate_retrieval_relevance(
    retrieved_documents,
    question,
):
    """
    Calculate a simple retrieval relevance score.

    The score measures how many retrieved documents
    contain at least one important keyword from the question.
    """

    if not retrieved_documents:
        return 0.0

    question_words = {
        word.lower().strip(".,?!")
        for word in question.split()
        if len(word) > 3
    }

    relevant_documents = 0

    for document in retrieved_documents:

        document_words = {
            word.lower().strip(".,?!")
            for word in document.page_content.split()
        }

        overlap = question_words.intersection(
            document_words
        )

        if overlap:
            relevant_documents += 1

    score = (
        relevant_documents
        / len(retrieved_documents)
    )

    return round(score, 3)


def calculate_citation_coverage(
    answer,
    retrieved_documents,
):
    """
    Measure how many retrieved sources are referenced
    in the generated answer.
    """

    if not retrieved_documents:
        return 0.0

    cited_sources = 0

    for document in retrieved_documents:

        source = document.metadata.get(
            "source",
            "",
        )

        page = str(
            document.metadata.get(
                "page",
                "",
            )
        )

        if source in answer and page in answer:

            cited_sources += 1

    score = (
        cited_sources
        / len(retrieved_documents)
    )

    return round(score, 3)


def calculate_groundedness(
    answer,
    retrieved_documents,
):
    """
    Estimate answer groundedness using word overlap
    between the answer and retrieved evidence.

    This is a lightweight heuristic metric.
    """

    if not answer or not retrieved_documents:
        return 0.0

    evidence_text = " ".join(
        document.page_content
        for document in retrieved_documents
    )

    answer_words = {
        word.lower().strip(".,?!")
        for word in answer.split()
        if len(word) > 4
    }

    evidence_words = {
        word.lower().strip(".,?!")
        for word in evidence_text.split()
        if len(word) > 4
    }

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(
        evidence_words
    )

    score = (
        len(overlap)
        / len(answer_words)
    )

    return round(
        min(score, 1.0),
        3,
    )


def evaluate_rag(
    question,
    answer,
    retrieved_documents,
):
    """
    Run all evaluation metrics for a RAG response.
    """

    retrieval_relevance = (
        calculate_retrieval_relevance(
            retrieved_documents,
            question,
        )
    )

    citation_coverage = (
        calculate_citation_coverage(
            answer,
            retrieved_documents,
        )
    )

    groundedness = (
        calculate_groundedness(
            answer,
            retrieved_documents,
        )
    )

    return {
        "retrieval_relevance": retrieval_relevance,
        "citation_coverage": citation_coverage,
        "groundedness": groundedness,
    }