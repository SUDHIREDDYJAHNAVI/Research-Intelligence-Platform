from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    """
    Create a local embedding model.

    The model runs entirely on the user's computer,
    so no OpenAI API credits are required.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )