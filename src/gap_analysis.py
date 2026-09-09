def analyze_research_gaps(llm, chunks, paper_names):
    """
    Identify limitations, unresolved problems, and potential
    research gaps across the uploaded research papers.
    """

    if not chunks:
        return "No research evidence was found."

    if not paper_names:
        return "No papers were provided."

    # Group chunks by paper
    papers = {
        paper_name: []
        for paper_name in paper_names
    }

    for chunk in chunks:

        source = chunk.get(
            "source",
            "Unknown"
        )

        if source in papers:
            papers[source].append(chunk)

    context_parts = []

    # Keep each paper separate
    for paper_name in paper_names:

        context_parts.append(
            f"\n========== PAPER: {paper_name} =========="
        )

        paper_chunks = papers[paper_name][:20]

        if not paper_chunks:

            context_parts.append(
                "No evidence available."
            )

            continue

        for chunk in paper_chunks:

            page = chunk.get(
                "page",
                "Unknown"
            )

            text = chunk.get(
                "text",
                ""
            )

            context_parts.append(
                f"[{paper_name}, Page {page}]\n"
                f"{text}"
            )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an academic research-gap analysis assistant.

Analyze ONLY the research papers provided below.

Your goal is to identify research limitations,
unresolved problems, and potential research gaps.

IMPORTANT RULES:

1. Use ONLY the supplied evidence.
2. Do not introduce outside research.
3. Do not invent limitations.
4. Clearly distinguish between:
   - limitations explicitly stated by authors
   - gaps that can reasonably be inferred from the evidence
5. Do not claim that something is a research gap unless
   the supplied evidence provides a reasonable basis.
6. Keep papers separate when discussing their findings.
7. Every important factual claim must include a page citation.
8. If evidence is insufficient, say:
   "Not specified in the retrieved evidence."

Provide the following sections:

### 1. Paper-Specific Limitations

For each paper, identify limitations explicitly
supported by the evidence.

### 2. Unresolved Problems

Identify problems that remain insufficiently addressed
by the uploaded papers.

### 3. Common Research Gaps

Identify gaps that appear across multiple papers.

### 4. Methodological Gaps

Identify areas where existing approaches could be
improved, based only on the evidence.

### 5. Potential Research Directions

Suggest realistic research directions that could
address the identified gaps.

Clearly label suggestions as potential future directions,
not established facts.

Use citations like:

[paper.pdf, Page 8]

RESEARCH PAPER EVIDENCE:

{context}

Produce a concise academic research-gap analysis.
"""

    response = llm.invoke(
        prompt
    )

    return response.content