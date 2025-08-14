import streamlit as st
from KG.chunk import load_neo4j_graph
from vectorRAG import query_vector_rag
from GraphRAG import generate_cypher_query

# Load graph and configs once
@st.cache_resource
def init_graph():
    return load_neo4j_graph()

graph, openAI_api, openAI_endpoint = init_graph()

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Battle: Vector vs Graph", page_icon="⚔️", layout="wide")
st.title("⚔️ RAG Battle Arena")
st.write("Compare **Vector RAG** vs **Graph RAG** answers side-by-side.")


# Input box
question = st.text_input("💬 Enter your question", placeholder="e.g., Who killed Napoleon?")

if st.button("Run Comparison", type="primary"):
    if question.strip():
        with st.spinner("⚡ Running Vector RAG..."):
            try:
                vector_answer = query_vector_rag(question=question)
            except Exception as e:
                vector_answer = f"❌ Error: {e}"

        with st.spinner("🧠 Running Graph RAG..."):
            try:
                graph_answer = generate_cypher_query(graph=graph, question=question)
            except Exception as e:
                graph_answer = f"❌ Error: {e}"

        # --- Layout for side-by-side answers ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Vector RAG")
            st.write(vector_answer if vector_answer else "_No answer returned_")

        with col2:
            st.subheader("🕸 Graph RAG")
            st.write(graph_answer if graph_answer else "_No answer returned_")

    else:
        st.warning("Please enter a question before running the comparison.")

# Optional styling
st.markdown(
    """
    <style>
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
