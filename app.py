import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from answer import (  # noqa: E402
    EMERGENCY_NOTE, HEALTH_NOTE, MODEL, SAFETY_NOTE, OllamaError, build_retriever, get_answer,
)
from retrieval import CHUNKS_PATH, load_chunks  # noqa: E402

st.set_page_config(page_title="JSJ Intelligence", page_icon="🖐️")


@st.cache_resource(show_spinner="Loading the document library and search models...")
def get_retriever():
    """Runs once per session of the app, not on every question."""
    retriever, method = build_retriever(load_chunks())
    return retriever, method


if not CHUNKS_PATH.exists():
    st.title("JSJ Intelligence")
    st.info(
        "This is a public preview. The assistant answers from a private library of copyrighted "
        "articles using a language model that runs on my own computer, and neither can be "
        "published here, so live questions are turned off. Below are screenshots of the app "
        "and its evaluation. The code and instructions to run it yourself are on GitHub."
    )
    shots = Path(__file__).parent / "docs" / "screenshots"
    for name, caption in [
        ("01_chat_answer.png", "A cited answer"),
        ("03_langchain_comparison.png", "My retrieval vs LangChain"),
        ("04_llamaindex_comparison.png", "My retrieval vs LlamaIndex"),
        ("05_independent_retrieval.png", "Retrieval on independent questions"),
    ]:
        if (shots / name).exists():
            st.image(str(shots / name), caption=caption)
    st.markdown("[View the code on GitHub](https://github.com/summer-waves/jsj-intelligence)")
    st.stop()

retriever, method = get_retriever()

st.title("JSJ Intelligence")
st.caption(
    "Ask about Jin Shin Jyutsu. Answers come only from a curated document library, with citations. "
    f"Search: {method}. Model: {MODEL} (runs locally)."
)
st.info(
    "This reports what the JSJ teachings in the library say. It is not medical advice and the "
    "library is not scientific evidence. If you have symptoms, please see a doctor."
)

with st.sidebar:
    st.header("Try these")
    examples = [
        "Which finger do I hold for fear?",
        "Where is Safety Energy Lock 25 located?",
        "What is the 36 conscious breaths exercise?",
        "Is Jin Shin Jyutsu a religion?",
        "What does the little finger represent?",
    ]
    picked = None
    for ex in examples:
        if st.button(ex, use_container_width=True):
            picked = ex
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.caption("Each question is answered on its own. The assistant does not remember earlier questions.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of result dicts from get_answer()


EXCERPT_WORDS = 40  # the page shows only a short excerpt; the full article is behind the link


def excerpt(text, n=EXCERPT_WORDS):
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def show_result(r):
    """Draw one answer, with all the notes and checks."""
    if r["emergency"]:
        st.error(EMERGENCY_NOTE)
    if r["health"]:
        st.warning(HEALTH_NOTE)
    if r["withheld"]:
        st.info(f"The model's answer was replaced by a safety check: {r['withheld']}.")
    st.write(r["answer"])
    if r["invalid"]:
        st.error(f"The answer cites source number(s) {r['invalid']} that do not exist. "
                 "Do not trust those citations.")
    if r["uncited"]:
        st.warning("These sentences have no citation and cannot be verified:\n\n"
                   + "\n".join(f"- {x}" for x in r["uncited"]))
    if r["no_citations"]:
        st.warning("The answer contains no citations, so it cannot be verified.")
    if not r["refused"]:
        with st.expander("Sources", expanded=False):
            for n, c in enumerate(r["results"], start=1):
                cited = " (cited in the answer)" if n in r["used"] else ""
                st.markdown(f"**[{n}] [{c['title']}]({c['url']})**{cited}  \n`{c['chunk_id']}`")
                st.caption(excerpt(c["text"]))
    st.caption(SAFETY_NOTE)


for r in st.session_state.history:
    with st.chat_message("user"):
        st.write(r["question"])
    with st.chat_message("assistant"):
        show_result(r)

typed = st.chat_input("Ask a question about Jin Shin Jyutsu")
question = picked or typed

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking... a local model on a laptop can take 10 to 60 seconds."):
                result = get_answer(question, retriever)
        except OllamaError as e:
            st.error(str(e))
        else:
            show_result(result)
            st.session_state.history.append(result)
