# fake_news_predictor.py
# -------------------------------------------------------
# Place this file in the same folder as:
#   - fake_news_model.keras
#   - tokenizer.pkl
# -------------------------------------------------------

import tensorflow as tf
import pickle
import re
import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# 1. Load Model & Tokenizer
# -------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model('fake_news_model.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("✅ Model ready!\n")

# -------------------------------------------------------
# 2. Cleaning (same as training — must be identical)
# -------------------------------------------------------
def professional_clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------------------------------
# 3. Predict Function
# -------------------------------------------------------
def predict_news(text):
    cleaned = professional_clean(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=300, padding='post', truncating='post')
    score   = model.predict(padded, verbose=0)[0][0]

    if score > 0.7:
        verdict = "🔴 FAKE  (high confidence)"
    elif score > 0.5:
        verdict = "🟠 FAKE  (low confidence)"
    elif score > 0.3:
        verdict = "🟡 REAL  (low confidence)"
    else:
        verdict = "🟢 REAL  (high confidence)"

    print(f"\n📰 Input  : {text[:80]}")
    print(f"🎯 Score  : {score:.4f}  (1 = fake, 0 = real)")
    print(f"📊 Verdict: {verdict}")
    print("-" * 60)

# -------------------------------------------------------
# 4. Interactive Loop — type your own news in terminal
# -------------------------------------------------------
print("=" * 60)
print("        FAKE NEWS DETECTOR — type 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\n📝 Paste a news headline or article: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("👋 Exiting.")
        break

    if len(user_input) < 10:
        print("⚠️  Too short — paste a proper headline or article.")
        continue

    predict_news(user_input)
