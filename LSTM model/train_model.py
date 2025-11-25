import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from data_preprocessor import prepare_fall_detection_data

# ---------------------------
# Build pure LSTM model
# ---------------------------
def build_lstm_model(time_steps, n_features):
    model = Sequential([
        LSTM(192, return_sequences=True, input_shape=(time_steps, n_features), recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.35),

        LSTM(128, return_sequences=True, recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.35),

        LSTM(96),
        Dropout(0.25),

        Dense(128, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00015,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model



# ---------------------------
# Find best threshold
# ---------------------------
def find_best_threshold(y_test, y_probs):
    best_t, best_f1 = 0.5, 0

    for t in np.arange(0.1, 0.9, 0.005):
        y_pred = (y_probs > t).astype(int)

        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))

        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"✅ Best Threshold using F1-Score: {best_t:.3f}")
    return best_t


# ---------------------------
# Train and evaluate
# ---------------------------
def train_and_evaluate(epochs=80):
    # Load data
    X_train, X_test, y_train, y_test = prepare_fall_detection_data(
        time_steps=60,
        step_size=2
    )
    if X_train is None:
        return

    # Class weights
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    # Build model
    model = build_lstm_model(X_train.shape[1], X_train.shape[2])
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0003, restore_best_weights=True,verbose=1)
    checkpoint = ModelCheckpoint("best_fall_detection_model.keras",
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint, lr_reduce]
    )

    # Best epoch
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\n✅ Training finished. Best epoch: {best_epoch}")

    # Evaluate
    loss, acc, prec, rec = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # Save final model
    model.save("fall_detection_model.keras")
    print("\n✅ Final model saved")

    # Training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.axvline(best_epoch-1, color='r', linestyle='--', label='Best Epoch')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.axvline(best_epoch-1, color='r', linestyle='--', label='Best Epoch')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    train_and_evaluate(epochs=80)
