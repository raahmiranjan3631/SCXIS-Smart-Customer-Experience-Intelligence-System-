# app.py
"""
SCXIS - Smart Customer Experience Intelligence System (Streamlit app)

Features:
- Live chat (session-based)
- Sentiment analysis (distilbert)
- Intent detection (zero-shot, lighter model)
- Emotion detection (distilroberta)
- CSAT collection and basic analytics dashboard
- CSV export of chat history
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ---------------------------
# 1) Model loading (CPU-safe)
# ---------------------------
@st.cache_resource
def load_pipelines():
    # Sentiment model (smaller, fast)
    sent_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sent_tok = AutoTokenizer.from_pretrained(sent_name)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sent_name)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=sent_model,
        tokenizer=sent_tok,
        device=-1,  # CPU only
    )

    # Intent (zero-shot) - lighter MNLI model
    intent_name = "valhalla/distilbart-mnli-12-1"
    intent_tok = AutoTokenizer.from_pretrained(intent_name)
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_name)
    intent_pipe = pipeline(
        "zero-shot-classification",
        model=intent_model,
        tokenizer=intent_tok,
        device=-1,
    )

    # Emotion detection - pretrained distilroberta based emotion model
    emotion_name = "j-hartmann/emotion-english-distilroberta-base"
    emotion_tok = AutoTokenizer.from_pretrained(emotion_name)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_name)
    emotion_pipe = pipeline(
        "text-classification",
        model=emotion_model,
        tokenizer=emotion_tok,
        return_all_scores=False,
        device=-1,
    )

    return sentiment_pipe, intent_pipe, emotion_pipe

# load models once (cached)
sentiment_pipe, intent_pipe, emotion_pipe = load_pipelines()

# ---------------------------
# 2) Candidate labels / config
# ---------------------------
CANDIDATE_INTENTS = ["complaint", "query", "feedback", "request", "greeting", "other"]

# ---------------------------
# 3) Session state init
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts {role, text, ts, sentiment, intent, emotion}
if "csat" not in st.session_state:
    st.session_state["csat"] = []  # list of ints
if "title" not in st.session_state:
    st.session_state["title"] = "SCXIS - Smart Customer Experience Intelligence System"

# ---------------------------
# 4) Helper functions
# ---------------------------
def analyze_sentiment(text: str):
    """Return label ('positive'|'negative'|'neutral') and score."""
    try:
        res = sentiment_pipe(text)[0]
        label = res["label"].lower()
        score = float(res.get("score", 0.0))
        if "positive" in label:
            return "positive", score
        elif "negative" in label:
            return "negative", score
        else:
            return "neutral", score
    except Exception:
        return "neutral", 0.0


def detect_intent(text: str, candidate_labels=CANDIDATE_INTENTS):
    """Return top intent label and confidence."""
    try:
        res = intent_pipe(text, candidate_labels)
        labels = res.get("labels", [])
        scores = res.get("scores", [])
        if labels:
            return labels[0], float(scores[0])
        return "other", 0.0
    except Exception:
        return "other", 0.0


def detect_emotion(text: str):
    """Return top emotion label (e.g., joy, anger, sadness)."""
    try:
        res = emotion_pipe(text)[0]
        label = res.get("label", "")
        score = float(res.get("score", 0.0))
        return label.lower(), score
    except Exception:
        return "neutral", 0.0


def append_message(role: str, text: str, sentiment=None, intent=None, emotion=None):
    st.session_state["messages"].append(
        {
            "role": role,
            "text": text,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sentiment": sentiment,
            "intent": intent,
            "emotion": emotion,
        }
    )


def compute_csat_stats():
    if not st.session_state["csat"]:
        return {"count": 0, "avg": None}
    vals = st.session_state["csat"]
    return {"count": len(vals), "avg": sum(vals) / len(vals)}


def messages_to_df():
    if not st.session_state["messages"]:
        return pd.DataFrame(columns=["role", "text", "timestamp", "sentiment", "intent", "emotion"])
    return pd.DataFrame(st.session_state["messages"])


def convert_df_to_csv_bytes(df: pd.DataFrame):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# 5) UI - Header & layout
# ---------------------------
st.set_page_config(page_title="SCXIS", page_icon="🤖", layout="wide")
st.title(st.session_state["title"])

left_col, right_col = st.columns([2, 1])

with right_col:
    st.markdown("**Quick Actions**")
    if st.button("Export chat CSV"):
        df = messages_to_df()
        if df.empty:
            st.warning("No messages to export.")
        else:
            csv_bytes = convert_df_to_csv_bytes(df)
            st.download_button(
                label="Download chat history CSV",
                data=csv_bytes,
                file_name="scxis_chat_history.csv",
                mime="text/csv",
            )
    st.markdown("---")
    csat_stats = compute_csat_stats()
    st.metric("CSAT Avg", f"{csat_stats['avg']:.2f}" if csat_stats["avg"] is not None else "N/A", delta=None)
    st.caption(f"CSAT Samples: {csat_stats['count']}")

# ---------------------------
# 6) Navigation
# ---------------------------
page = st.sidebar.radio("Navigation", ["Live Chat", "Dashboard", "Chat History", "Settings"])

# ---------------------------
# 7) Live Chat page
# ---------------------------
if page == "Live Chat":
    st.header("💬 Live Chat")
    st.markdown("Interact with the system. The AI analyzes sentiment, intent, and emotion in real-time.")

    # Chat input area
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input("Type your message", key="user_input")
        submitted = st.form_submit_button("Send")
        if submitted and user_text:
            # analyze
            sent_label, sent_score = analyze_sentiment(user_text)
            intent_label, intent_score = detect_intent(user_text)
            emotion_label, emotion_score = detect_emotion(user_text)

            # append user
            append_message("user", user_text, sentiment=sent_label, intent=intent_label, emotion=emotion_label)

            # generate a simple response (rule-based + placeholders)
            if intent_label == "complaint" and sent_label == "negative":
                bot_text = "😟 I'm sorry you're facing this issue. I'll escalate this to our support team."
            elif intent_label == "query":
                bot_text = "❓ Thanks — I can help with that. Could you share a bit more detail?"
            elif intent_label == "feedback":
                bot_text = "💡 Thank you for the feedback! We value it."
            elif intent_label == "greeting":
                bot_text = "👋 Hello! How can I assist you today?"
            else:
                # tone aware default
                if sent_label == "positive":
                    bot_text = "🙂 Great to hear! Anything else I can help with?"
                elif sent_label == "negative":
                    bot_text = "I understand. Can you share an order/issue ID so I can check?"
                else:
                    bot_text = "Thanks for sharing — I'll note this."

            append_message("bot", bot_text, sentiment=None, intent=None, emotion=None)
            st.experimental_rerun()

    # show recent messages
    st.subheader("Conversation")
    msgs = st.session_state["messages"][-20:]
    for m in msgs:
        ts = m["timestamp"]
        if m["role"] == "user":
            st.markdown(f"**You** ({ts}):  {m['text']}")
            st.caption(f"Sentiment: {m['sentiment'] or '-'} | Intent: {m['intent'] or '-'} | Emotion: {m['emotion'] or '-'}")
        else:
            st.markdown(f"**Bot** ({ts}):  {m['text']}")

    # collect CSAT optionally
    with st.expander("Provide CSAT (optional)"):
        csat = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 5, 4, key="csat_slider")
        if st.button("Submit CSAT"):
            st.session_state["csat"].append(int(csat))
            st.success("Thanks — your rating has been recorded.")

# ---------------------------
# 8) Dashboard page
# ---------------------------
elif page == "Dashboard":
    st.header("📊 Insights Dashboard")
    st.markdown("Overview of conversation analytics and trends.")

    df = messages_to_df()
    user_df = df[df["role"] == "user"] if not df.empty else pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        if not user_df.empty:
            sent_counts = user_df["sentiment"].value_counts()
            st.bar_chart(sent_counts)
        else:
            st.info("No user messages yet.")

    with col2:
        st.subheader("Intent Distribution")
        if not user_df.empty:
            intent_counts = user_df["intent"].value_counts()
            st.bar_chart(intent_counts)
        else:
            st.info("No user messages yet.")

    st.subheader("Emotion Distribution")
    if not user_df.empty:
        emo_counts = user_df["emotion"].value_counts()
        st.bar_chart(emo_counts)
    else:
        st.info("No user messages yet.")

    st.markdown("---")
    st.subheader("CSAT Trend")
    if st.session_state["csat"]:
        st.line_chart(pd.Series(st.session_state["csat"], name="CSAT"))
    else:
        st.info("No CSAT data yet.")

    st.markdown("---")
    st.subheader("Recent Key Insights")
    if not user_df.empty:
        # Simple trend: top intents and any spike of negatives
        top_intent = user_df["intent"].mode().iloc[0] if not user_df["intent"].empty else "N/A"
        neg_pct = (user_df["sentiment"] == "negative").sum() / max(1, len(user_df)) * 100
        st.write(f"- Top intent this session: **{top_intent}**")
        st.write(f"- Negative messages: **{neg_pct:.1f}%** of user messages")
    else:
        st.write("No insights yet — invite users to chat.")

# ---------------------------
# 9) Chat History page
# ---------------------------
elif page == "Chat History":
    st.header("📜 Chat History")
    df_all = messages_to_df()
    if df_all.empty:
        st.info("No chat history yet.")
    else:
        # show table and allow filters
        st.dataframe(df_all.sort_values(by="timestamp", ascending=False).reset_index(drop=True))
        csv_bytes = convert_df_to_csv_bytes(df_all)
        st.download_button("Download full chat CSV", csv_bytes, "scxis_full_chat_history.csv", "text/csv")

# ---------------------------
# 10) Settings page
# ---------------------------
elif page == "Settings":
    st.header("⚙️ Settings & About")
    st.markdown(
        """
        **SCXIS** — Smart Customer Experience Intelligence System  
        Components: Chat interface, Sentiment, Intent, Emotion, CSAT, Dashboard.  
        Models: DistilBERT (sentiment), DistilBART (intent), DistilRoBERTa (emotion).
        """
    )
    st.markdown("Model loading is cached to improve performance. For production, consider persistent DB storage and authentication.")
