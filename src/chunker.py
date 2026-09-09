from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages: list[dict]) -> list[dict]:
    """
    Split extracted PDF pages into smaller chunks while
    preserving source and page metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for page in pages:
        text_chunks = splitter.split_text(page["text"])

        for chunk_index, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_index + 1,
                }
            )

    return chunks