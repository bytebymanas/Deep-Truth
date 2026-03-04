import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. Saved Model load karo
model = tf.keras.models.load_model('RIYAL-V3-PRO.keras')

# 2. Saved Tokenizer load karo
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def check_news(title, text):
    # Title aur Text ko merge karo (jaise training mein kiya tha)
    full_news = title + " " + text
    
    # Text ko numbers mein badlo
    seq = tokenizer.texts_to_sequences([full_news])
    padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')
    
    # Prediction lo
    prediction = model.predict(padded)
    
    # Result dikhao
    prob = prediction[0][0]
    if prob > 0.5:
        print(f"\nResult: FAKE NEWS! ❌ (Confidence: {prob*100:.2f}%)")
    else:
        print(f"\nResult: REAL NEWS! ✅ (Confidence: {(1-prob)*100:.2f}%)")


t = str(input("Enter News Title: "))
tx = str(input("Enter News Description: "))
check_news(t, tx)