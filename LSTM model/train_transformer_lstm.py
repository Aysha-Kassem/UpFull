# ============================================
# train_transformer_lstm.py
# ============================================
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_preprocessor_transformer import prepare_transformer_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# -------------------------------
# Transformer Encoder Block
# -------------------------------
def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation="relu")(x)
    return x + res

# -------------------------------
# Build Transformer + LSTM Hybrid
# -------------------------------
def build_transformer_lstm(time_steps, features):
    inp = Input(shape=(time_steps, features))
    
    # LSTM part
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    
    # Transformer Encoder
    x = transformer_encoder(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    
    shared = layers.Dense(128, activation="relu")(x)
    shared = layers.Dropout(0.3)(shared)

    # Outputs
    fall_now = layers.Dense(64, activation="relu")(shared)
    fall_now = layers.Dense(1, activation="sigmoid", name="fall_now")(fall_now)

    fall_soon = layers.Dense(64, activation="relu")(shared)
    fall_soon = layers.Dense(1, activation="sigmoid", name="fall_soon")(fall_soon)

    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# -------------------------------
# Train
# -------------------------------
def train_transformer_model():
    X_train, X_test, y_train_now, y_test_now, y_train_soon, y_test_soon = prepare_transformer_data()

    time_steps = X_train.shape[1]
    features = X_train.shape[2]

    model = build_transformer_lstm(time_steps, features)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4),
        ModelCheckpoint("BEST_TRANSFORMER_LSTM.keras", monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train,
        {"fall_now": y_train_now, "fall_soon": y_train_soon},
        validation_data=(X_test, {"fall_now": y_test_now, "fall_soon": y_test_soon}),
        epochs=60,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    model.save("FINAL_TRANSFORMER_LSTM.keras")
    return model, history, (X_test, y_test_now, y_test_soon)

# -------------------------------
# Plot history & confusion matrices
# -------------------------------
def plot_results(history, X_test, y_test_now, y_test_soon, model):
    plt.figure(figsize=(12,6))
    plt.plot(history.history["fall_now_accuracy"])
    plt.plot(history.history["val_fall_now_accuracy"])
    plt.title("Fall Now Accuracy")
    plt.legend(["train","val"])
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(history.history["fall_soon_accuracy"])
    plt.plot(history.history["val_fall_soon_accuracy"])
    plt.title("Fall Soon Accuracy")
    plt.legend(["train","val"])
    plt.show()

    # Predictions & Confusion Matrix
    y_pred_now, y_pred_soon = model.predict(X_test)
    y_pred_now = (y_pred_now > 0.5).astype(int)
    y_pred_soon = (y_pred_soon > 0.5).astype(int)

    # Fall Now
    print("=== Fall Now Classification Report ===")
    print(classification_report(y_test_now, y_pred_now))
    cm_now = confusion_matrix(y_test_now, y_pred_now)
    ConfusionMatrixDisplay(cm_now).plot(cmap="Blues")
    plt.title("Fall Now Confusion Matrix")
    plt.show()

    # Fall Soon
    print("=== Fall Soon Classification Report ===")
    print(classification_report(y_test_soon, y_pred_soon))
    cm_soon = confusion_matrix(y_test_soon, y_pred_soon)
    ConfusionMatrixDisplay(cm_soon).plot(cmap="Oranges")
    plt.title("Fall Soon Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    model, history, test_data = train_transformer_model()
    plot_results(history, *test_data, model)
