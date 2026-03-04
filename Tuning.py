# this file is used for HyperParameter Tuning

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras import regularizers
import keras_tuner as kt
import pickle

# --- STEP 1: DATA LOADING & PREP ---

df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna()
df['full_news'] = df['title'] + " " + df['text']

y = df['label'].values   

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['full_news'])
sequences = tokenizer.texts_to_sequences(df['full_news'])

# Padding (300 length for consistency)
padded_X = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')

# --- STEP 2: BUILD MODEL FUNCTION ---
def build_model(hp):
    model = Sequential([
        # Masking ON taaki zeros ignore ho jayein (PM Modi error fix)
        Embedding(10000, 16, mask_zero=True),
        GlobalAveragePooling1D(),
        
        # Hyperparameter: Neurons (16 to 64)
        Dense(units=hp.Int('units', 16, 64, step=16), 
              activation='relu', 
              kernel_regularizer=regularizers.l2(0.01)),
        
        # Hyperparameter: Dropout (0.2 to 0.5)
        Dropout(rate=hp.Float('dropout_rate', 0.2, 0.5, step=0.1)),
        
        Dense(12, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Hyperparameter: Learning Rate
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- STEP 3: TUNER SETUP ---
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='tuning_logs',
    project_name='FakeNews_Final_Phase1'
)

# Early Stopping to save time
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# --- STEP 4: START SEARCH ---
print("Searching for the best hyperparameters... M1 GPU ON! 🚀")
tuner.search(padded_X, y, epochs=10, validation_split=0.2, callbacks=[stop_early])

# Get the results
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
\n--- PHASE 1 TUNING COMPLETE ---
Best Units: {best_hps.get('units')}
Best Dropout Rate: {best_hps.get('dropout_rate')}
Best Learning Rate: {best_hps.get('learning_rate')}
""")