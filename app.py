# ============================================================
# 🤖 Sentiment + Sarcasm Detection App (Streamlit)
# ============================================================

import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# ------------------------------------------------------------
# 🧩 Streamlit Config
# ------------------------------------------------------------
st.set_page_config(page_title="Sentiment + Sarcasm Analyzer", page_icon="🤖")
st.write("✅ Streamlit loaded! Models are initializing... please wait a few seconds.")

# ------------------------------------------------------------
# 🔹 Load Local Sentiment Model (Your Fine-Tuned Roberta)
# ------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    local_model_path = r"C:\Users\Siddharth\OneDrive\Desktop\NLP\roberta_sentiment"  # ✅ your fine-tuned Amazon sentiment model

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False
    )
    return sentiment_pipeline

# ------------------------------------------------------------
# 🔹 Load Sarcasm / Irony Model (Hugging Face)
# ------------------------------------------------------------
@st.cache_resource
def load_sarcasm_model():
    model_name = "cardiffnlp/twitter-roberta-base-irony"
    sarcasm_pipeline = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    return sarcasm_pipeline

# Initialize both models
sentiment_pipeline = load_sentiment_model()
sarcasm_pipeline = load_sarcasm_model()

# ------------------------------------------------------------
# 🧠 Text Input
# ------------------------------------------------------------
st.title("🎭 Sentiment + Sarcasm Classifier")
text_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            # ---------------------------
            # Sentiment Prediction (5-class)
            # ---------------------------
            sentiment_result = sentiment_pipeline(text_input)[0]

            # Handle model output (label_0 ... label_4)
            label_text = sentiment_result["label"]
            sentiment_score = float(sentiment_result["score"])

            try:
                label_id = int(label_text.split("_")[-1]) if "_" in label_text else int(label_text)
            except:
                label_id = 0  # fallback if parsing fails

            # Map label IDs (0–4) to readable stars
            sentiment_map = {
                0: "⭐ Terrible",
                1: "⭐⭐ Bad",
                2: "⭐⭐⭐ Neutral",
                3: "⭐⭐⭐⭐ Good",
                4: "⭐⭐⭐⭐⭐ Excellent"
            }
            sentiment_display = sentiment_map.get(label_id, "Unknown")

            # ---------------------------
            # Sarcasm (Irony) Prediction
            # ---------------------------
            sarcasm_result = sarcasm_pipeline(text_input)[0]
            sarcasm_label = sarcasm_result["label"].lower()  # 'irony' or 'not irony'
            sarcasm_score = float(sarcasm_result["score"])

            # ---------------------------
            # Combined Interpretation Logic 🎯
            # ---------------------------
            st.subheader("Results:")
            st.write(f"**Sentiment:** {sentiment_display}")
            st.write(f"**Confidence:** `{sentiment_score:.2f}`")
            st.write(f"\n**Sarcasm Label:** {sarcasm_label}")
            st.write(f"**Score:** `{sarcasm_score:.2f}`")

            # Sentiment + sarcasm combined logic
            if sarcasm_label == "irony":
                st.error(f"⚠️ Sarcasm detected! (Confidence: {sarcasm_score:.2f})")

                # If the sentiment is positive but sarcastic
                if label_id in [3, 4]:
                    st.warning(
                        "🤔 Although the sentiment appears **positive**, sarcasm was detected — "
                        "**the overall tone is not genuinely positive.**"
                    )
            else:
                st.success(f"✅ No sarcasm detected. (Confidence: {sarcasm_score:.2f})")

# ------------------------------------------------------------
# ℹ️ Notes
# ------------------------------------------------------------
st.caption("""
**Models Used:**
- 🧠 Sentiment: *Your Custom Fine-Tuned Roberta (Amazon Reviews, 5-class)*
- 🎭 Sarcasm (Irony): `cardiffnlp/twitter-roberta-base-irony`

✅ Works offline once cached  
⚙️ Automatically uses GPU if available
""")
