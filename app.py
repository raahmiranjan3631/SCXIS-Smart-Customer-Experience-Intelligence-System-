# app.py
"""
SCXIS - Smart Customer Experience Intelligence System (Streamlit app)
Patched & improved:
 - Uses Hugging Face Inference API (no local torch required)
 - Robust duplicate-submission guarding (last_processed + processing flag)
 - Heuristic fallbacks if HF API or token is absent
 - Live chat, sentiment, intent, emotion, CSAT, dashboard, CSV export
Note: Add HF_API_TOKEN in Streamlit Secrets (key name: HF_API_TOKEN)
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import requests

# ---------------------------
# Config / Secrets
# ---------------------------
st.set_page_config(page_title="SCXIS", page_icon="🤖", layout="wide")

# Read HF token from Streamlit secrets (safe storage)
HF_TOKEN = st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else None
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Candidate intents
CANDIDATE_INTENTS = ["complaint", "query", "feedback", "request", "greeting", "other"]

# ---------------------------
# Session state init
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts
if "csat" not in st.session_state:
    st.session_state["csat"] = []
if "title" not in st.session_state:
    st.session_state["title"] = "SCXIS - Smart Customer Experience Intelligence System"
if "last_processed" not in st.session_state:
    st.session_state["last_processed"] = ""  # last processed user text
if "processing" not in st.session_state:
    st.session_state["processing"] = False
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""  # keeps input persistent across reruns

# ---------------------------
# HF Inference helpers
# ---------------------------
HF_BASE = "https://api-inference.huggingface.co/models"

def hf_infer(model_id: str, payload: dict, timeout: int = 30):
    """Call HF inference API and return parsed JSON. Raises on HTTP error."""
    url = f"{HF_BASE}/{model_id}"
    r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def analyze_sentiment_hf(text: str):
    model = "distilbert-base-uncased-finetuned-sst-2-english"
    out = hf_infer(model, {"inputs": text})
    if isinstance(out, list) and out:
        label = out[0].get("label", "").lower()
        score = float(out[0].get("score", 0.0))
        if "positive" in label:
            return "positive", score
        if "negative" in label:
            return "negative", score
    return "neutral", 0.0

def detect_intent_hf(text: str, candidate_labels=CANDIDATE_INTENTS):
    model = "valhalla/distilbart-mnli-12-1"
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    out = hf_infer(model, payload)
    labels = out.get("labels", [])
    scores = out.get("scores", [])
    if labels:
        return labels[0], float(scores[0])
    return "other", 0.0

def detect_emotion_hf(text: str):
    model = "j-hartmann/emotion-english-distilroberta-base"
    out = hf_infer(model, {"inputs": text})
    if isinstance(out, list) and out:
        label = out[0].get("label", "").lower()
        score = float(out[0].get("score", 0.0))
        return label, score
    if isinstance(out, dict):
        items = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
        if items:
            return items[0][0].lower(), float(items[0][1])
    return "neutral", 0.0

# ---------------------------
# Simple heuristics (fallbacks)
# ---------------------------
NEG_WORDS = {"angry","upset","terrible","worst","hate","refund","broken","not working","issue","problem","delay"}
POS_WORDS = {"good","great","love","awesome","happy","satisfied","thank","thanks","excellent","perfect"}
CONF_WORDS = {"how","how do","what","why","can't","cannot","unable","confused","unclear"}

def heuristic_sentiment(text: str):
    t = text.lower()
    neg = sum(1 for w in NEG_WORDS if w in t)
    pos = sum(1 for w in POS_WORDS if w in t)
    conf = sum(1 for w in CONF_WORDS if w in t)
    if neg > max(pos, conf):
        return "negative", 0.6
    if pos > neg:
        return "positive", 0.7
    return "neutral", 0.0

def heuristic_intent(text: str):
    t = text.lower()
    if any(k in t for k in ["refund","charge","billing","payment"]):
        return "complaint"
    if any(k in t for k in ["how","what","why","where","help","assist","support","can you", "unable"]):
        return "query"
    if any(k in t for k in ["feedback","suggestion","recommend"]):
        return "feedback"
    if any(k in t for k in ["hi","hello","hey","good morning","good evening"]):
        return "greeting"
    return "other"

def heuristic_emotion(text: str):
    t = text.lower()
    if any(w in t for w in ["angry","furious","mad","irritat"]):
        return "anger", 0.7
    if any(w in t for w in ["happy","thank","glad","love","great"]):
        return "joy", 0.8
    if any(w in t for w in ["sad","disappointed","unhappy"]):
        return "sadness", 0.6
    return "neutral", 0.0

# ---------------------------
# Utility functions
# ---------------------------
def append_message(role: str, text: str, sentiment=None, intent=None, emotion=None):
    st.session_state["messages"].append({
        "role": role,
        "text": text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sentiment": sentiment,
        "intent": intent,
        "emotion": emotion
    })

def messages_to_df():
    if not st.session_state["messages"]:
        return pd.DataFrame(columns=["role","text","timestamp","sentiment","intent","emotion"])
    return pd.DataFrame(st.session_state["messages"])

def convert_df_to_csv_bytes(df: pd.DataFrame):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# UI - Header & layout
# ---------------------------
st.title(st.session_state["title"])

left_col, right_col = st.columns([2,1])
with right_col:
    st.markdown("**Quick Actions**")
    st.markdown("")
    if st.button("Export chat CSV"):
        df = messages_to_df()
        if df.empty:
            st.warning("No messages to export.")
        else:
            csv_bytes = convert_df_to_csv_bytes(df)
            st.download_button("Download chat history CSV", data=csv_bytes, file_name="scxis_chat_history.csv", mime="text/csv")
    st.markdown("---")
    csat_count = len(st.session_state["csat"])
    csat_avg = (sum(st.session_state["csat"]) / csat_count) if csat_count else None
    st.metric("CSAT Avg", f"{csat_avg:.2f}" if csat_avg is not None else "N/A")
    st.caption(f"CSAT Samples: {csat_count}")
    st.markdown("---")
    if not HF_TOKEN:
        st.warning("HF_API_TOKEN missing in Streamlit secrets — using heuristics as fallback. Add HF_API_TOKEN to enable HF Inference API.")

# ---------------------------
# Navigation
# ---------------------------
page = st.sidebar.radio("Navigation", ["Live Chat", "Dashboard", "Chat History", "Settings"])

# ---------------------------
# Live Chat
# ---------------------------
if page == "Live Chat":
    st.header("💬 Live Chat")
    st.markdown("Type a message — the system analyzes sentiment, intent, and emotion.")

    # Safer form that prevents duplicate processing
    with st.form("chat_form", clear_on_submit=False):
        st.session_state["user_input"] = st.text_input("Type your message", value=st.session_state["user_input"], key="user_input_widget")
        submitted = st.form_submit_button("Send")

        if submitted:
            user_text = st.session_state["user_input"].strip()

            if st.session_state["processing"]:
                st.warning("Still processing previous message — please wait.")
            elif not user_text:
                st.warning("Please enter a message.")
            elif user_text == st.session_state["last_processed"]:
                st.info("This message was just processed; please enter a new message.")
            else:
                # start processing
                st.session_state["processing"] = True
                try:
                    # Use HF API when token available; fall back to heuristics on errors
                    try:
                        sent_label, sent_score = analyze_sentiment_hf(user_text) if HF_TOKEN else heuristic_sentiment(user_text)
                    except Exception:
                        sent_label, sent_score = heuristic_sentiment(user_text)

                    try:
                        intent_label, intent_score = detect_intent_hf(user_text, CANDIDATE_INTENTS) if HF_TOKEN else (heuristic_intent(user_text), 0.0)
                    except Exception:
                        intent_label, intent_score = heuristic_intent(user_text), 0.0

                    try:
                        emotion_label, emotion_score = detect_emotion_hf(user_text) if HF_TOKEN else heuristic_emotion(user_text)
                    except Exception:
                        emotion_label, emotion_score = heuristic_emotion(user_text)

                    # append user (only once)
                    append_message("user", user_text, sentiment=sent_label, intent=intent_label, emotion=emotion_label)

                    # bot response (simple rule-based)
                    if intent_label == "complaint" and sent_label == "negative":
                        bot_text = "😟 I'm sorry you're facing this issue. I'll escalate this to our support team."
                    elif intent_label == "query":
                        bot_text = "❓ Thanks — I can help with that. Could you share a bit more detail?"
                    elif intent_label == "feedback":
                        bot_text = "💡 Thank you for the feedback! We value it."
                    elif intent_label == "greeting":
                        bot_text = "👋 Hello! How can I assist you today?"
                    else:
                        if sent_label == "positive":
                            bot_text = "🙂 Great to hear! Anything else I can help with?"
                        elif sent_label == "negative":
                            bot_text = "I understand. Can you share an order/issue ID so I can check?"
                        else:
                            bot_text = "Thanks for sharing — I'll note this."

                    append_message("bot", bot_text)

                    # update guards
                    st.session_state["last_processed"] = user_text
                    st.session_state["user_input"] = ""  # clear input
                    st.success("Message processed.")
                except Exception as e:
                    st.error(f"Processing error: {e}")
                finally:
                    st.session_state["processing"] = False

    # show conversation
    st.subheader("Conversation (most recent)")
    msgs = st.session_state["messages"][-40:]
    for m in msgs:
        ts = m["timestamp"]
        if m["role"] == "user":
            st.markdown(f"**You** ({ts}): {m['text']}")
            st.caption(f"Sentiment: {m['sentiment'] or '-'} | Intent: {m['intent'] or '-'} | Emotion: {m['emotion'] or '-'}")
        else:
            st.markdown(f"**Bot** ({ts}): {m['text']}")

    # CSAT
    with st.expander("Provide CSAT (optional)"):
        csat = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 5, 4, key="csat_slider")
        if st.button("Submit CSAT"):
            st.session_state["csat"].append(int(csat))
            st.success("Thanks — your rating has been recorded.")

# ---------------------------
# Dashboard
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
            st.bar_chart(user_df["sentiment"].value_counts())
        else:
            st.info("No user messages yet.")

    with col2:
        st.subheader("Intent Distribution")
        if not user_df.empty:
            st.bar_chart(user_df["intent"].value_counts())
        else:
            st.info("No user messages yet.")

    st.subheader("Emotion Distribution")
    if not user_df.empty:
        st.bar_chart(user_df["emotion"].value_counts())
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
        top_intent = user_df["intent"].mode().iloc[0] if not user_df["intent"].empty else "N/A"
        neg_pct = (user_df["sentiment"] == "negative").sum() / max(1, len(user_df)) * 100
        st.write(f"- Top intent this session: **{top_intent}**")
        st.write(f"- Negative messages: **{neg_pct:.1f}%** of user messages")
    else:
        st.write("No insights yet — invite users to chat.")

# ---------------------------
# Chat History
# ---------------------------
elif page == "Chat History":
    st.header("📜 Chat History")
    df_all = messages_to_df()
    if df_all.empty:
        st.info("No chat history yet.")
    else:
        st.dataframe(df_all.sort_values(by="timestamp", ascending=False).reset_index(drop=True))
        csv_bytes = convert_df_to_csv_bytes(df_all)
        st.download_button("Download full chat CSV", csv_bytes, "scxis_full_chat_history.csv", "text/csv")

# ---------------------------
# Settings
# ---------------------------
elif page == "Settings":
    st.header("⚙️ Settings & About")
    st.markdown(
        """
        **SCXIS** — Smart Customer Experience Intelligence System  
        Components: Chat interface, Sentiment, Intent, Emotion, CSAT, Dashboard.  
        This deployment uses the Hugging Face Inference API to avoid heavy local dependencies.
        """
    )
    st.markdown("Notes:")
    st.markdown("- Set `HF_API_TOKEN` in Streamlit Secrets to enable HF Inference API. Without it the app uses heuristics.")
    st.markdown("- HF Inference API may add latency on first calls; consider caching repeated calls if needed.")
