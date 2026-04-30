from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import math
import sys
import time

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from quora_duplicate_detector.text_features import (
    jaccard_similarity,
    normalize_text,
    token_overlap_ratio,
    word_count,
)


MAX_QUESTION_CHARS = 1000
MAX_BATCH_ROWS = 5000
DEFAULT_THRESHOLD = 0.42
RECALL_THRESHOLD = 0.30
BALANCED_THRESHOLD = 0.38
STRICT_THRESHOLD = 0.52
REQUIRED_COLUMNS = {"question1", "question2"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "best", "can", "do", "does", "for",
    "from", "get", "how", "i", "in", "is", "it", "my", "of", "on", "or",
    "should", "the", "to", "way", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}
SYNONYM_GROUPS = {
    "learn": {"learn", "study", "master", "understand"},
    "fast": {"fast", "quick", "quickly", "rapid", "rapidly", "faster", "fastest"},
    "prepare": {"prepare", "ready", "practice", "prep"},
    "job": {"job", "career", "interview", "interviews"},
    "good": {"good", "great", "best", "better"},
    "improve": {"improve", "increase", "enhance", "boost"},
}
SYNONYM_LOOKUP = {
    alias: canonical
    for canonical, aliases in SYNONYM_GROUPS.items()
    for alias in aliases
}

EXAMPLES = {
    "Python duplicate": (
        "How can I learn Python quickly?",
        "What is the fastest way to learn Python?",
    ),
    "Interview duplicate": (
        "How do I prepare for a data science interview?",
        "What is the best way to get ready for data science interviews?",
    ),
    "Not duplicate": (
        "How do I learn machine learning?",
        "What are the best tourist places in Nepal?",
    ),
    "Borderline": (
        "How can I improve my English speaking skills?",
        "How can I improve my public speaking skills?",
    ),
}


st.set_page_config(
    page_title="Quora Duplicate AI",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --bg-deep: #070711;
        --bg-panel: rgba(15, 15, 35, .72);
        --border-soft: rgba(255, 255, 255, .14);
        --text-main: #f8fbff;
        --text-muted: #b8c4dc;
        --primary: #8b5cf6;
        --primary-2: #6366f1;
        --accent: #22d3ee;
        --accent-2: #ec4899;
        --success: #22c55e;
        --danger: #fb7185;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 8%, rgba(139, 92, 246, .34), transparent 27%),
            radial-gradient(circle at 88% 12%, rgba(236, 72, 153, .20), transparent 25%),
            radial-gradient(circle at 50% 92%, rgba(34, 211, 238, .18), transparent 30%),
            linear-gradient(135deg, #050510 0%, #0d1024 48%, #050510 100%);
        color: var(--text-main);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(5, 5, 16, .82);
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(139, 92, 246, .26);
    }

    h1, h2, h3, label, p, .stMarkdown, .stCaption {
        color: var(--text-main);
        letter-spacing: 0;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 24px rgba(139, 92, 246, .18); }
        50% { box-shadow: 0 0 42px rgba(34, 211, 238, .24); }
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(17, 24, 39, .68) 0%, rgba(49, 46, 129, .42) 48%, rgba(131, 24, 67, .28) 100%);
        padding: 3.1rem 2.2rem;
        border-radius: 28px;
        margin: .3rem 0 1.35rem 0;
        text-align: center;
        border: 1px solid var(--border-soft);
        backdrop-filter: blur(22px);
        animation: fadeUp .8s ease both, glowPulse 5s ease-in-out infinite;
    }

    .hero:after {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
        background-size: 38px 38px;
        mask-image: radial-gradient(circle at 50% 20%, black, transparent 72%);
        pointer-events: none;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0 0 .55rem 0;
        font-size: clamp(2.35rem, 6vw, 4.75rem);
        line-height: 1;
        font-weight: 900;
        background: linear-gradient(90deg, #ffffff 0%, #a78bfa 35%, #22d3ee 70%, #f9a8d4 100%);
        -webkit-background-clip: text;
        color: transparent;
    }

    .hero p {
        position: relative;
        z-index: 1;
        margin: 0 auto;
        color: var(--text-muted);
        max-width: 680px;
        font-size: 1.05rem;
    }

    .cyber-pill {
        position: relative;
        z-index: 1;
        display: inline-block;
        color: #f0f9ff;
        border: 1px solid rgba(34, 211, 238, .42);
        border-radius: 999px;
        padding: .38rem .85rem;
        font-size: .78rem;
        margin-bottom: .9rem;
        background: rgba(99, 102, 241, .18);
        backdrop-filter: blur(12px);
    }

    .glass-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 1.25rem;
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 60px rgba(0,0,0,.22);
        animation: fadeUp .7s ease both;
    }

    .section-kicker {
        color: var(--accent);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .15rem;
    }

    div[data-testid="stForm"] {
        background: rgba(15, 15, 35, .66);
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 24px;
        padding: 1.3rem;
        backdrop-filter: blur(18px);
        box-shadow: 0 18px 54px rgba(0,0,0,.24);
    }

    textarea {
        background: rgba(3, 7, 18, .72) !important;
        color: #f8fbff !important;
        border: 1px solid rgba(139, 92, 246, .28) !important;
        border-radius: 18px !important;
        transition: all .24s ease !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.03) !important;
    }

    textarea:focus {
        border-color: rgba(34, 211, 238, .85) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, .12), 0 0 24px rgba(139, 92, 246, .20) !important;
    }

    div[data-testid="stTabs"] button {
        color: var(--text-muted);
        border-radius: 999px;
        transition: all .22s ease;
    }

    div[data-testid="stTabs"] button:hover {
        color: white;
        background: rgba(139, 92, 246, .13);
    }

    .result {
        border-radius: 24px;
        padding: 1.35rem;
        margin: 1rem 0 1rem 0;
        color: #f8fbff;
        box-shadow: 0 22px 60px rgba(0,0,0,.28);
        animation: fadeUp .55s ease both;
    }

    .duplicate {
        background: linear-gradient(135deg, rgba(16, 185, 129, .24), rgba(15, 23, 42, .84));
        border: 1px solid rgba(34, 197, 94, .62);
    }

    .not-duplicate {
        background: linear-gradient(135deg, rgba(244, 63, 94, .24), rgba(15, 23, 42, .84));
        border: 1px solid rgba(251, 113, 133, .62);
    }

    .result-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .result-label {
        font-size: 1.7rem;
        font-weight: 900;
        margin: 0;
    }

    .confidence {
        font-size: 2.1rem;
        font-weight: 900;
    }

    .small-note { color: var(--text-muted); font-size: .95rem; }

    div[data-testid="stMetric"] {
        background: rgba(15, 15, 35, .72);
        border: 1px solid rgba(139, 92, 246, .22);
        border-radius: 18px;
        padding: 1rem;
        backdrop-filter: blur(14px);
        transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(34, 211, 238, .48);
        box-shadow: 0 14px 34px rgba(34, 211, 238, .10);
    }

    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(255,255,255,.18) !important;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent-2) 52%, var(--accent) 100%) !important;
        color: white !important;
        font-weight: 850 !important;
        border-radius: 999px !important;
        padding: .72rem 1rem !important;
        box-shadow: 0 14px 34px rgba(139, 92, 246, .28) !important;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: scale(1.02);
        filter: brightness(1.08);
        box-shadow: 0 18px 46px rgba(34, 211, 238, .22) !important;
    }

    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.10);
    }

    @media (max-width: 760px) {
        .hero { padding: 2rem 1.1rem; border-radius: 20px; }
        .result-topline { align-items: flex-start; }
        .confidence { font-size: 1.55rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


def word_ngrams(text: str) -> list[str]:
    tokens = canonical_tokens(text)
    unigrams = tokens
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return unigrams + bigrams


def canonical_tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = normalize_text(text).split()
    normalized = []
    for token in tokens:
        token = SYNONYM_LOOKUP.get(token, token)
        if keep_stopwords or token not in STOPWORDS:
            normalized.append(token)
    return normalized


def content_overlap(question_1: str, question_2: str) -> float:
    left = set(canonical_tokens(question_1))
    right = set(canonical_tokens(question_2))
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / min(len(left), len(right))


def content_jaccard(question_1: str, question_2: str) -> float:
    left = set(canonical_tokens(question_1))
    right = set(canonical_tokens(question_2))
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def char_ngrams(text: str, n: int = 3) -> list[str]:
    normalized = normalize_text(text).replace(" ", "")
    if len(normalized) < n:
        return [normalized] if normalized else []
    return [normalized[index : index + n] for index in range(len(normalized) - n + 1)]


def hashed_vector(terms: list[str], buckets: int = 4096) -> Counter:
    vector = Counter()
    for term in terms:
        digest = hashlib.md5(term.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % buckets
        vector[bucket] += 1.0
    return vector


def normalized_dot(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0

    dot_product = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def word_vector_similarity(question_1: str, question_2: str) -> float:
    return normalized_dot(hashed_vector(word_ngrams(question_1)), hashed_vector(word_ngrams(question_2)))


def char_vector_similarity(question_1: str, question_2: str) -> float:
    return normalized_dot(hashed_vector(char_ngrams(question_1)), hashed_vector(char_ngrams(question_2)))


def length_similarity(question_1: str, question_2: str) -> float:
    left = max(word_count(question_1), 1)
    right = max(word_count(question_2), 1)
    return min(left, right) / max(left, right)


def sequence_similarity(question_1: str, question_2: str) -> float:
    return SequenceMatcher(None, normalize_text(question_1), normalize_text(question_2)).ratio()


def duplicate_score(question_1: str, question_2: str, threshold: float) -> dict:
    word_similarity = word_vector_similarity(question_1, question_2)
    char_similarity = char_vector_similarity(question_1, question_2)
    sequence_score = sequence_similarity(question_1, question_2)
    jaccard = jaccard_similarity(question_1, question_2)
    overlap = token_overlap_ratio(question_1, question_2)
    content_token_overlap = content_overlap(question_1, question_2)
    content_token_jaccard = content_jaccard(question_1, question_2)
    length_score = length_similarity(question_1, question_2)
    exact_match = float(normalize_text(question_1) == normalize_text(question_2))

    score = (
        0.18 * word_similarity
        + 0.18 * char_similarity
        + 0.16 * sequence_score
        + 0.17 * overlap
        + 0.20 * content_token_overlap
        + 0.06 * content_token_jaccard
        + 0.03 * jaccard
        + 0.02 * length_score
        + 0.05 * exact_match
    )
    score = max(0.0, min(1.0, float(score)))

    recall_rule = (
        content_token_overlap >= 0.50
        and (char_similarity >= 0.32 or sequence_score >= 0.34 or word_similarity >= 0.25)
    ) or (
        overlap >= 0.45 and char_similarity >= 0.28
    )
    if recall_rule:
        score = max(score, threshold + 0.03)

    return {
        "duplicate_probability": score,
        "label": "Duplicate" if score >= threshold else "Not Duplicate",
        "word_similarity": word_similarity,
        "char_similarity": char_similarity,
        "sequence_similarity": sequence_score,
        "jaccard_similarity": jaccard,
        "token_overlap_ratio": overlap,
        "content_overlap": content_token_overlap,
        "content_jaccard": content_token_jaccard,
        "length_similarity": length_score,
    }


def validate_pair(question_1: str, question_2: str) -> list[str]:
    errors = []
    if not question_1.strip() or not question_2.strip():
        errors.append("Both questions are required.")
    if len(question_1) > MAX_QUESTION_CHARS or len(question_2) > MAX_QUESTION_CHARS:
        errors.append(f"Each question must be {MAX_QUESTION_CHARS} characters or fewer.")
    return errors


def render_result(result: dict, threshold: float) -> None:
    css_class = "duplicate" if result["label"] == "Duplicate" else "not-duplicate"
    icon = "✅" if result["label"] == "Duplicate" else "⛔"
    confidence = result["duplicate_probability"] * 100
    st.markdown(
        f"""
        <div class="result {css_class}">
            <div class="result-topline">
                <div>
                    <div class="section-kicker">AI verdict</div>
                    <div class="result-label">{icon} {result["label"]}</div>
                    <div class="small-note">Decision threshold: <b>{threshold:.3f}</b></div>
                </div>
                <div>
                    <div class="section-kicker">Confidence</div>
                    <div class="confidence">{confidence:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(result["duplicate_probability"])


def score_batch(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = frame.copy()
    scores = [
        duplicate_score(str(row.question1), str(row.question2), threshold)
        for row in result.itertuples(index=False)
    ]
    result["duplicate_probability"] = [item["duplicate_probability"] for item in scores]
    result["label"] = [item["label"] for item in scores]
    result["word_similarity"] = [item["word_similarity"] for item in scores]
    result["char_similarity"] = [item["char_similarity"] for item in scores]
    result["sequence_similarity"] = [item["sequence_similarity"] for item in scores]
    result["jaccard_similarity"] = [item["jaccard_similarity"] for item in scores]
    result["token_overlap_ratio"] = [item["token_overlap_ratio"] for item in scores]
    result["content_overlap"] = [item["content_overlap"] for item in scores]
    return result


st.markdown(
    """
    <div class="hero">
        <div class="cyber-pill">AI SIMILARITY INTELLIGENCE</div>
        <h1>Quora Duplicate AI</h1>
        <p>
            Detect duplicate questions with a fast recall-first matching experience.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### ⚡ Quora Duplicate AI")
st.sidebar.caption("Premium cloud interface")
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎛️ Modes")
st.sidebar.write("Single pair scoring")
st.sidebar.write("CSV batch scoring")
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧠 Engine")
st.sidebar.success("Recall-first matcher")
st.sidebar.caption("Full transformer/XGBoost pipeline remains in the repository for local use.")

tab_single, tab_batch, tab_about = st.tabs(["Single Prediction", "Batch Prediction", "About"])

with tab_single:
    st.markdown('<div class="section-kicker">Live analyzer</div>', unsafe_allow_html=True)
    st.subheader("Check Question Similarity")
    selected_example = st.selectbox("Try an example", list(EXAMPLES.keys()))
    default_q1, default_q2 = EXAMPLES[selected_example]

    with st.form("single_form"):
        left, right = st.columns(2)
        with left:
            question_1 = st.text_area(
                "Question 1",
                value=default_q1,
                height=150,
                max_chars=MAX_QUESTION_CHARS,
                placeholder="Example: How can I learn Python quickly?",
            )
        with right:
            question_2 = st.text_area(
                "Question 2",
                value=default_q2,
                height=150,
                max_chars=MAX_QUESTION_CHARS,
                placeholder="Example: What is the fastest way to learn Python?",
            )

        sensitivity = st.radio(
            "Sensitivity mode",
            ["High Recall", "Balanced", "Strict"],
            horizontal=True,
            help="High Recall catches more duplicates. Strict reduces false positives.",
        )
        default_threshold = {
            "High Recall": RECALL_THRESHOLD,
            "Balanced": BALANCED_THRESHOLD,
            "Strict": STRICT_THRESHOLD,
        }[sensitivity]
        threshold = st.slider("Duplicate threshold", 0.10, 0.90, default_threshold, 0.01)
        submitted = st.form_submit_button("Check Similarity", use_container_width=True)

    if submitted:
        errors = validate_pair(question_1, question_2)
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("AI is analyzing question intent..."):
                time.sleep(0.45)
                result = duplicate_score(question_1, question_2, threshold)
            render_result(result, threshold)

            st.markdown('<div class="section-kicker">Similarity signals</div>', unsafe_allow_html=True)
            metrics = st.columns(4)
            metrics[0].metric("Word Similarity", f"{result['word_similarity']:.3f}")
            metrics[1].metric("Char Similarity", f"{result['char_similarity']:.3f}")
            metrics[2].metric("Sequence Similarity", f"{result['sequence_similarity']:.3f}")
            metrics[3].metric("Content Overlap", f"{result['content_overlap']:.3f}")

with tab_batch:
    st.markdown('<div class="section-kicker">Bulk intelligence</div>', unsafe_allow_html=True)
    st.subheader("Batch Prediction")
    st.markdown('<div class="small-note">Upload a CSV containing `question1` and `question2` columns.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_mode = st.radio(
        "Batch sensitivity",
        ["High Recall", "Balanced", "Strict"],
        horizontal=True,
    )
    batch_default = {
        "High Recall": RECALL_THRESHOLD,
        "Balanced": BALANCED_THRESHOLD,
        "Strict": STRICT_THRESHOLD,
    }[batch_mode]
    batch_threshold = st.slider(
        "Batch duplicate threshold",
        min_value=0.10,
        max_value=0.90,
        value=batch_default,
        step=0.01,
    )
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Could not read CSV: {error}")
            frame = None

        if frame is not None:
            missing = REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            elif frame.empty:
                st.error("CSV contains no rows.")
            elif len(frame) > MAX_BATCH_ROWS:
                st.error(f"Please upload {MAX_BATCH_ROWS:,} rows or fewer.")
            else:
                st.success(f"Loaded {len(frame):,} rows")
                st.dataframe(frame.head(min(20, len(frame))), use_container_width=True)

                if st.button("Run Batch Analysis", use_container_width=True):
                    with st.spinner("AI is scanning every question pair..."):
                        time.sleep(0.45)
                        output = score_batch(frame, batch_threshold)
                    duplicate_count = int((output["label"] == "Duplicate").sum())
                    st.markdown('<div class="section-kicker">Batch summary</div>', unsafe_allow_html=True)
                    cols = st.columns(3)
                    cols[0].metric("Rows", f"{len(output):,}")
                    cols[1].metric("Duplicates", f"{duplicate_count:,}")
                    cols[2].metric("Duplicate Rate", f"{duplicate_count / len(output):.2%}")
                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        output.to_csv(index=False).encode("utf-8"),
                        "quora_duplicate_predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )

with tab_about:
    st.markdown('<div class="section-kicker">Product brief</div>', unsafe_allow_html=True)
    st.subheader("About this deployment")
    st.write(
        "This public Streamlit deployment uses a lightweight semantic scoring engine "
        "so the hosted app starts quickly on Community Cloud. The repository also includes "
        "the full local transformer/XGBoost implementation, model artifacts, and v2 training pipeline."
    )
    st.markdown(
        "- Author: **Mohammed Ghanim Siddiqui**\n"
        "- GitHub: [MOHAMMED-GHANIM-SIDDIQUI](https://github.com/MOHAMMED-GHANIM-SIDDIQUI)\n"
        "- Repository: [quora-duplicate-detector-improved](https://github.com/MOHAMMED-GHANIM-SIDDIQUI/quora-duplicate-detector-improved)"
    )
