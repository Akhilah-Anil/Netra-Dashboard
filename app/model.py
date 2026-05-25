"""
model.py
========
LSTM Autoencoder for per-segment anomaly detection.

Input  : (batch, max_len, 17)  — 1 raw signal + 16 stat features
Output : (batch, max_len, 17)  — reconstructed sequence

Architecture
------------
Encoder : LSTM(128, seq=True) → Dropout(0.2) → LSTM(latent, seq=False)
Bridge  : RepeatVector(max_len)
Decoder : LSTM(latent, seq=True) → Dropout(0.2) → LSTM(128, seq=True)
Output  : TimeDistributed(Dense(17))

Loss    : MSE across all timesteps and features.
          The raw signal (feature 0) has the most temporal variance and
          dominates the reconstruction error at inference time. The 16 stat
          features act as a regularising context that tightens the normal
          manifold in latent space.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    RepeatVector, TimeDistributed,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_lstm_autoencoder(max_len: int,
                            n_features: int = 17,
                            latent_dim: int = 64) -> Model:
    """
    Build and compile the LSTM Autoencoder.

    Parameters
    ----------
    max_len    : fixed sequence length (e.g. 100)
    n_features : number of input features per timestep (17 = 1 signal + 16 stats)
    latent_dim : bottleneck LSTM units (default 64)

    Returns
    -------
    Compiled Keras Model
    """
    inp = Input(shape=(max_len, n_features), name='input')

    # ── Encoder ──────────────────────────────────────────────────────────
    x = LSTM(256, activation='tanh', return_sequences=True,
             name='enc_lstm1')(inp)
    x = Dropout(0.30, name='enc_dropout')(x)
    x = LSTM(latent_dim, activation='tanh', return_sequences=False,
             name='enc_lstm2_bottleneck')(x)   # → (batch, latent_dim)

    # ── Bridge ────────────────────────────────────────────────────────────
    x = RepeatVector(max_len, name='repeat')(x)  # → (batch, max_len, latent_dim)

    # ── Decoder ──────────────────────────────────────────────────────────
    x = LSTM(latent_dim, activation='tanh', return_sequences=True,
             name='dec_lstm1')(x)
    x = Dropout(0.30, name='dec_dropout')(x)
    x = LSTM(256, activation='tanh', return_sequences=True,
             name='dec_lstm2')(x)
    out = TimeDistributed(Dense(n_features), name='output')(x)

    model = Model(inp, out, name='lstm_autoencoder')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
    )
    return model


def get_callbacks(patience_stop: int = 8,
                  patience_lr: int = 4) -> list:
    """
    EarlyStopping + ReduceLROnPlateau on val_loss.

    patience_stop : epochs with no improvement before stopping
    patience_lr   : epochs with no improvement before halving LR
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
