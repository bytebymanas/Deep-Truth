import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers


# 1. Load Data
df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna()
df['full_news'] = df['title'] + " " + df['text']

# 2. Labels
y = df['label'].values   

# 3. Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['full_news'])
sequences = tokenizer.texts_to_sequences(df['full_news'])

# 4. Padding
padded_X = pad_sequences(
    sequences, maxlen=300, padding='post', truncating='post'
)

# 5. Model
model = Sequential([
    Embedding(10000, 16 , mask_zero=True),
    GlobalAveragePooling1D(),
    
    # Hidden Layer with Dropout & Regularization
    Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5), # 50% neurons randomly off honge training ke waqt
    
    # Ek aur Layer (Optional, for better depth)
    Dense(12, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Early Stopping setup
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# 6. TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.fit(
    padded_X, y,
    epochs=20, 
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback, early_stop]
)

# 1. Model ko save karo (H5 format mein)
model.save('RIYAL-V1.keras')

# Tokenizer save karo (pickle use karke)
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
