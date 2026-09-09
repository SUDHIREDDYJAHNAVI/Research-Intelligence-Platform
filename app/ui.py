import sys
from pathlib import Path

import streamlit as st

# Allow imports from the src directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import create_embedding_model
from src.retriever import build_vector_store, retrieve_relevant_chunks
from src.rag import create_llm, generate_answer
from src.comparison import compare_papers
from src.gap_analysis import analyze_research_gaps
from src.evaluation import evaluate_rag


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    embedding_model = create_embedding_model()
    llm = create_llm()
    return embedding_model, llm


# =========================================================
# PROCESS PAPERS
# =========================================================

def process_papers(uploaded_files, embedding_model):

    all_pages = []

    temp_dir = ROOT_DIR / "data" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:

        file_path = temp_dir / uploaded_file.name

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        pages = extract_text_from_pdf(str(file_path))
        all_pages.extend(pages)

    chunks = create_chunks(all_pages)

    vector_store = build_vector_store(
        chunks,
        embedding_model,
    )

    return vector_store, chunks


# =========================================================
# HEADER
# =========================================================

st.title("🧠 Research Intelligence Platform")

st.markdown(
    """
### AI-powered research paper analysis

Upload research papers and use **semantic retrieval, local LLMs,
evidence-grounded question answering, paper comparison, and
research-gap analysis**.
"""
)

st.divider()


# =========================================================
# LOAD MODELS
# =========================================================

embedding_model, llm = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧠 Research Intelligence")

st.sidebar.markdown(
    """
**AI Research Assistant**

Analyze multiple research papers using
local NLP and Retrieval-Augmented Generation.
"""
)

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Technology Stack")

st.sidebar.markdown(
    """
- 🐍 Python
- 🤗 Sentence Transformers
- 🔎 FAISS
- 🧠 Llama 3.2
- 📚 RAG
- 📄 PDF Processing
- 📊 Streamlit
"""
)

st.sidebar.divider()

st.sidebar.markdown("### ✨ Capabilities")

st.sidebar.markdown(
    """
- Multi-PDF processing
- Semantic search
- Evidence-grounded Q&A
- Page-level citations
- Multi-paper comparison
- Research-gap analysis
- RAG evaluation metrics
"""
)


# =========================================================
# UPLOAD PAPERS
# =========================================================

st.subheader("📚 Upload Research Papers")

uploaded_files = st.file_uploader(
    "Select one or more PDF research papers",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload research papers in PDF format.",
)


# =========================================================
# PROCESS PAPERS
# =========================================================

if uploaded_files:

    st.write(
        f"**{len(uploaded_files)} paper(s) selected**"
    )

    with st.expander("📄 Selected Papers", expanded=True):

        for file in uploaded_files:
            st.write(f"• {file.name}")

    if st.button(
        "🚀 Process Research Papers",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Processing papers, creating chunks and building the vector database..."
        ):

            try:

                vector_store, chunks = process_papers(
                    uploaded_files,
                    embedding_model,
                )

                st.session_state["vector_store"] = vector_store
                st.session_state["chunks"] = chunks

                st.session_state["paper_names"] = [
                    file.name
                    for file in uploaded_files
                ]

                st.success(
                    f"Successfully processed "
                    f"{len(uploaded_files)} paper(s) "
                    f"and created "
                    f"{len(chunks)} searchable chunks."
                )

            except Exception as error:

                st.error(
                    f"Processing failed: {error}"
                )


# =========================================================
# PROJECT STATUS
# =========================================================

if "vector_store" in st.session_state:

    st.divider()

    st.subheader("📈 Project Status")

    paper_names = st.session_state.get(
        "paper_names",
        [],
    )

    chunks = st.session_state.get(
        "chunks",
        [],
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Research Papers",
            len(paper_names),
        )

    with col2:
        st.metric(
            "Searchable Chunks",
            len(chunks),
        )

    with col3:
        st.metric(
            "AI Model",
            "Llama 3.2",
        )


# =========================================================
# MAIN ANALYSIS TABS
# =========================================================

if "vector_store" in st.session_state:

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "🔎 Ask AI",
            "📊 Compare Papers",
            "🧩 Research Gaps",
        ]
    )


    # =====================================================
    # TAB 1 — ASK AI
    # =====================================================

    with tab1:

        st.subheader(
            "🔎 Ask Your Research Question"
        )

        st.write(
            "Ask questions about the uploaded papers. "
            "The system retrieves relevant evidence and "
            "generates an answer using the local Llama 3.2 model."
        )

        question = st.text_area(
            "Research Question",
            placeholder=(
                "Example: What datasets were used in these "
                "papers and what were their main limitations?"
            ),
            height=120,
        )

        if st.button(
            "🤖 Ask AI",
            type="primary",
            use_container_width=True,
        ):

            if not question.strip():

                st.warning(
                    "Please enter a research question."
                )

            else:

                with st.spinner(
                    "Searching research evidence and generating answer..."
                ):

                    try:

                        documents = retrieve_relevant_chunks(
                            st.session_state["vector_store"],
                            question,
                            k=5,
                        )

                        result = generate_answer(
                            llm,
                            documents,
                            question,
                        )

                        st.divider()

                        st.subheader(
                            "📑 Evidence-Grounded Answer"
                        )

                        st.write(
                            result["answer"]
                        )


                        # =================================
                        # EVALUATION
                        # =================================

                        metrics = evaluate_rag(
                            question,
                            result["answer"],
                            documents,
                        )

                        st.divider()

                        st.subheader(
                            "📊 RAG Evaluation"
                        )

                        st.caption(
                            "Lightweight heuristic indicators based on "
                            "retrieved evidence and answer overlap."
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Retrieval Relevance",
                                f"{metrics['retrieval_relevance']:.1%}",
                            )

                        with col2:

                            st.metric(
                                "Citation Coverage",
                                f"{metrics['citation_coverage']:.1%}",
                            )

                        with col3:

                            st.metric(
                                "Groundedness",
                                f"{metrics['groundedness']:.1%}",
                            )


                        # =================================
                        # SOURCES
                        # =================================

                        st.divider()

                        st.subheader(
                            "📚 Retrieved Sources"
                        )

                        seen_sources = set()

                        for source in result["sources"]:

                            source_key = (
                                source["source"],
                                source["page"],
                            )

                            if source_key not in seen_sources:

                                st.write(
                                    f"📄 **{source['source']}**  "
                                    f"— Page {source['page']}"
                                )

                                seen_sources.add(
                                    source_key
                                )

                    except Exception as error:

                        st.error(
                            f"Unable to generate answer: {error}"
                        )


    # =====================================================
    # TAB 2 — COMPARE PAPERS
    # =====================================================

    with tab2:

        st.subheader(
            "📊 Compare Research Papers"
        )

        st.write(
            "Compare the objectives, methodologies, datasets, "
            "findings, and limitations of the uploaded papers."
        )

        paper_names = st.session_state.get(
            "paper_names",
            [],
        )

        if len(paper_names) < 2:

            st.info(
                "Upload and process at least 2 research papers "
                "to enable multi-paper comparison."
            )

        else:

            st.markdown(
                "**Papers available for comparison:**"
            )

            for paper in paper_names:
                st.write(f"• {paper}")

            st.write("")

            if st.button(
                "🔬 Compare Papers",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Analyzing and comparing the research papers..."
                ):

                    try:

                        comparison = compare_papers(
                            llm,
                            st.session_state["chunks"],
                            paper_names,
                        )

                        st.divider()

                        st.subheader(
                            "📑 Paper Comparison"
                        )

                        st.write(
                            comparison
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to compare papers: {error}"
                        )


    # =====================================================
    # TAB 3 — RESEARCH GAP ANALYSIS
    # =====================================================

    with tab3:

        st.subheader(
            "🧩 Research Gap Analysis"
        )

        st.write(
            "Identify reported limitations, unresolved problems, "
            "methodological gaps, and potential future research "
            "directions from the uploaded papers."
        )

        paper_names = st.session_state.get(
            "paper_names",
            [],
        )

        if not paper_names:

            st.info(
                "Upload and process research papers first."
            )

        else:

            if st.button(
                "🧠 Analyze Research Gaps",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Analyzing research limitations and potential gaps..."
                ):

                    try:

                        gap_analysis = analyze_research_gaps(
                            llm,
                            st.session_state["chunks"],
                            paper_names,
                        )

                        st.divider()

                        st.subheader(
                            "🧩 Research Gap Analysis"
                        )

                        st.write(
                            gap_analysis
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to analyze research gaps: "
                            f"{error}"
                        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Research Intelligence Platform • "
    "Local NLP + FAISS + Retrieval-Augmented Generation + Llama 3.2"
)