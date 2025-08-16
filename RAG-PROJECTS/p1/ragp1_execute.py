# gui.py
import streamlit as st
from app import build_pipeline  # import your pipeline builder

# --------------------------
# Setup
# --------------------------
st.set_page_config(page_title="RAG Q&A Bot", page_icon="🤖", layout="centered")

st.title("🤖 Local Document Q&A Bot")
st.caption("Context-aware answers using your local documents + Mistral")

# Initialize pipeline only once (and cache it)
@st.cache_resource(show_spinner=True)
def load_pipeline():
    return build_pipeline(force_rebuild=False)

qa = load_pipeline()

# --------------------------
# UI
# --------------------------
user_input = st.text_area("Ask a question about your document:", height=100)

if st.button("Run Query"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = qa.invoke({"query": user_input})
            answer = result.get("result", "").strip()

        st.subheader("Answer")
        st.write(answer)

        with st.expander("🔎 Show retrieved context"):
            for i, doc in enumerate(result.get("source_documents", []), 1):
                src = doc.metadata.get("source", "unknown")
                preview = doc.page_content.strip().replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                st.markdown(f"**{i}. {src}**\n\n{preview}")
    else:
        st.warning("Please type a question first.")
