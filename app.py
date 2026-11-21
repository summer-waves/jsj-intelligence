import base64
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from answer import (  # noqa: E402
    ANTHROPIC_KEY, DEMO, EMERGENCY_NOTE, HEALTH_NOTE, MODEL, SAFETY_NOTE, OllamaError,
    build_retriever, get_answer,
)
from retrieval import load_chunks  # noqa: E402

st.set_page_config(page_title="JSJ Intelligence", page_icon="🖐️")


def add_background():
    """Use assets/background.jpg (or .jpeg/.png) as the page background. Does nothing if the file is missing."""
    assets = Path(__file__).resolve().parent / "assets"
    for name, mime in (("background.jpg", "jpeg"), ("background.jpeg", "jpeg"), ("background.png", "png")):
        path = assets / name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/{mime};base64,{data}");
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                }}
                [data-testid="stMainBlockContainer"], .block-container {{
                    background: rgba(255, 255, 255, 0.85);
                    border-radius: 16px;
                    padding: 2rem 2.5rem 3rem 2.5rem;
                    margin-top: 1.5rem;
                    margin-bottom: 1.5rem;
                }}
                [data-testid="stHeader"] {{ background: transparent; }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            return


add_background()


@st.cache_resource(show_spinner="Loading the document library and search models...")
def get_retriever():
    """Runs once per session of the app, not on every question."""
    retriever, method = build_retriever(load_chunks())
    return retriever, method


retriever, method = get_retriever()

if DEMO:
    st.title("Mind-Body Library Assistant (demo)")
    st.caption(
        "A public demo of my RAG assistant. Answers come only from 5 public NIH pages (NCCIH) on relaxation, "
        "stress, anxiety, yoga and massage, with citations. "
        f"Search: {method}. Model: {MODEL} (hosted)."
    )
    st.info(
        "This is a demo. My JSJ Intelligence project answers from a private library of Jin Shin Jyutsu "
        "articles that I cannot publish, so this version uses public-domain NIH pages instead. "
        "It summarizes what those pages say and is not medical advice. If you have symptoms, please see a doctor."
    )
else:
    st.title("JSJ Intelligence")
    where = "hosted" if ANTHROPIC_KEY else "runs locally"
    st.caption(
        "Ask about Jin Shin Jyutsu. Answers come only from a curated document library, with citations. "
        f"Search: {method}. Model: {MODEL} ({where})."
    )
    st.info(
        "This reports what the JSJ teachings in the library say. It is not medical advice and the "
        "library is not scientific evidence. If you have symptoms, please see a doctor."
    )

with st.sidebar:
    st.header("Try these")
    if DEMO:
        examples = [
            "What are relaxation techniques?",
            "Can yoga help with anxiety?",
            "What are the risks of massage therapy?",
            "Does deep breathing lower blood pressure?",
            "Which finger do I hold for fear?",
        ]
    else:
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

typed = st.chat_input(
    "Ask a question about relaxation, stress, yoga or massage" if DEMO else "Ask a question about Jin Shin Jyutsu"
)
question = picked or typed

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..." if ANTHROPIC_KEY else "Thinking... a local model on a laptop can take 10 to 60 seconds."):
                result = get_answer(question, retriever)
        except OllamaError as e:
            st.error(str(e))
        else:
            show_result(result)
            st.session_state.history.append(result)
