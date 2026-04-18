import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout

# --- STEP 1: Professional Cleaning Function ---
def professional_clean(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove Punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove Numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    import gc
    # Data load aur clean karne ke baad extra memory release karo
    del df # Agar dataframe ki ab zaroorat nahi hai
    gc.collect()


# --- STEP 2: Data Loading & Global Shuffling ---
print("🚀 Loading and Shuffling Data...")
df = pd.read_csv('WELFake_Dataset.csv').dropna()
df['full_news'] = df['title'] + " " + df['text']

# Global Shuffle: Model ko "ratta" maarne se rokne ke liye
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Apply Cleaning
print("🧹 Cleaning Text (Professional Style)...")
df['full_news'] = df['full_news'].apply(professional_clean)

X = df['full_news'].values
y = df['label'].values

# --- STEP 3: Tokenization & Padding ---
vocab_size = 10000
max_length = 300 # M1 ke liye 300 safe hai with batch 64

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded_X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# --- STEP 4: Professional Split ---
X_train, X_val, y_train, y_val = train_test_split(padded_X, y, test_size=0.2, random_state=42)

# --- STEP 5: The BEAST Model Architecture ---
model = Sequential([
    # Step C: Hero Step (Embedding)
    Embedding(vocab_size, 16, mask_zero=True), 
    
    # Sequence Processing
    LSTM(32, activation='tanh', recurrent_activation='sigmoid'), 
    
    # Classification Head
    Dense(16, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# M1/M2 Macs ke liye fast aur stable version
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# --- STEP 6: The Safety Callbacks ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=2, 
    min_lr=0.00001, 
    verbose=1
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/beast_v2")

# --- STEP 7: Training ---
print("🔥 Training Started (Beast Mode ON)...")
model.fit(
    X_train, y_train,
    epochs=20, 
    batch_size=64, # Optimized for M1 GPU
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, tensorboard_callback]
)

# --- STEP 8: Saving Everything ---
model.save('BEAST_LSTM_V2.keras')
with open('tokenizer_v2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model and Tokenizer saved! Mission Accomplished.")