def compare_papers(llm, chunks, paper_names):
    """
    Compare uploaded research papers using only their
    own extracted chunks.

    Each paper is summarized independently first.
    The summaries are then compared.
    """

    if not chunks:
        return "No research evidence was found."

    if not paper_names:
        return "No papers were provided."

    # -----------------------------------------------------
    # Group chunks by paper
    # -----------------------------------------------------

    papers = {}

    for paper_name in paper_names:
        papers[paper_name] = []

    for chunk in chunks:

        source = chunk.get(
            "source",
            "Unknown"
        )

        if source in papers:
            papers[source].append(chunk)

    # -----------------------------------------------------
    # Create an independent summary for each paper
    # -----------------------------------------------------

    paper_summaries = {}

    for paper_name in paper_names:

        paper_chunks = papers.get(
            paper_name,
            []
        )

        if not paper_chunks:
            paper_summaries[paper_name] = (
                "No evidence was retrieved for this paper."
            )
            continue

        # Limit context to keep local Llama reasonably fast
        selected_chunks = paper_chunks[:20]

        evidence_parts = []

        for chunk in selected_chunks:

            page = chunk.get(
                "page",
                "Unknown"
            )

            text = chunk.get(
                "text",
                ""
            )

            evidence_parts.append(
                f"[{paper_name}, Page {page}]\n"
                f"{text}"
            )

        evidence = "\n\n".join(
            evidence_parts
        )

        summary_prompt = f"""
You are analyzing ONE research paper.

Paper name:
{paper_name}

Use ONLY the evidence from this paper.

Do not use outside knowledge.
Do not mention other papers.
Do not invent information.

Extract:

1. Main objective
2. Main methodology
3. Dataset / experimental setup
4. Key findings
5. Reported limitations

Every factual statement must include a page citation.

If information is not present, write:

"Not specified in the retrieved evidence."

PAPER EVIDENCE:

{evidence}

Create a concise academic summary.
"""

        summary_response = llm.invoke(
            summary_prompt
        )

        paper_summaries[paper_name] = (
            summary_response.content
        )

    # -----------------------------------------------------
    # Build comparison context
    # -----------------------------------------------------

    comparison_parts = []

    for paper_name in paper_names:

        comparison_parts.append(
            f"========== {paper_name} ==========\n"
            f"{paper_summaries[paper_name]}"
        )

    comparison_context = "\n\n".join(
        comparison_parts
    )

    # -----------------------------------------------------
    # Compare the independent summaries
    # -----------------------------------------------------

    comparison_prompt = f"""
You are an academic research comparison assistant.

You are given independent summaries of the uploaded
research papers.

Compare ONLY these papers.

IMPORTANT RULES:

1. Do not introduce papers that were not uploaded.
2. Do not mention RoBERTa, DistilBERT, ELMo, or other
   models unless they appear in the supplied summaries.
3. Do not invent information.
4. Do not transfer methodology from one paper to another.
5. Preserve the citations provided in the summaries.
6. Do not declare one paper better unless there is
   clear evidence for that conclusion.
7. If information is missing, say:
   "Not specified in the retrieved evidence."

Provide:

### Paper-by-Paper Summary

For each paper:
- Objective
- Methodology
- Dataset / Experimental Setup
- Key Findings
- Limitations

### Similarities

Only similarities supported by the summaries.

### Differences

Focus on objective, methodology, datasets,
experiments, and findings.

### Overall Comparison

Explain how the papers differ and what types of
problems each approach addresses.

Do not compare them with unrelated external literature.

PAPER SUMMARIES:

{comparison_context}

Produce a concise and academically useful comparison.
"""

    final_response = llm.invoke(
        comparison_prompt
    )

    return final_response.content