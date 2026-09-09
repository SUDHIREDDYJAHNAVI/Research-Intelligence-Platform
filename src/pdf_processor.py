from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        A list of dictionaries containing page number and extracted text.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                    "source": path.name,
                }
            )

    return pages