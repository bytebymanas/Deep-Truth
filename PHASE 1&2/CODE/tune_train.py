import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras import regularizers

# 1. Load Data
df = pd.read_csv('WELFake_Dataset.csv').dropna()
df['full_news'] = df['title'] + " " + df['text']

# 2. Tokenization & Padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['full_news'])
sequences = tokenizer.texts_to_sequences(df['full_news'])
padded_X = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')

# 3. Final Architecture (Using Tuner's Result)
model = Sequential([
    Embedding(10000, 16, mask_zero=True), # FIXED: Masking ON
    GlobalAveragePooling1D(),
    Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.02)), # Units: 16
    Dropout(0.4), # Dropout: 0.4
    Dense(12, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile with Best LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Full Training
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Final Model Training Start... 🚀")
model.fit(padded_X, df['label'].values, epochs=20, validation_split=0.2, callbacks=[early_stop])

# 6. Save the Legend
model.save('RIYAL-RNN.keras')
print("Phase 1 Successfully Completed! Model Saved as RIYAL-RNN.keras")