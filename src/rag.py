from langchain_ollama import ChatOllama


def create_llm():
    """
    Create the local Llama 3.2 language model through Ollama.
    """
    return ChatOllama(
        model="llama3.2",
        temperature=0,
    )


def generate_answer(llm, retrieved_documents, question: str):
    """
    Generate an evidence-grounded answer using retrieved
    research-paper content.
    """

    if not retrieved_documents:
        return {
            "answer": "I could not find relevant evidence in the uploaded papers.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for document in retrieved_documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{document.page_content}"
        )

        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a research intelligence assistant.

Answer the user's question using ONLY the research-paper
evidence provided below.

Rules:
1. Do not invent facts.
2. If the evidence is insufficient, say so.
3. Synthesize information across papers when appropriate.
4. Give a clear academic answer.
5. Cite important claims using [Source, Page].

RESEARCH PAPER EVIDENCE:

{context}

USER QUESTION:

{question}

Provide an evidence-grounded answer with citations.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }


if __name__ == "__main__":
    print("Testing local Llama 3.2...")

    llm = create_llm()

    response = llm.invoke(
        "Say exactly: Research Intelligence Platform local AI is working."
    )

    print(response.content)