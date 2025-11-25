import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessor import prepare_fall_detection_data

# --------------------------------
# Build Advanced BiLSTM model
# --------------------------------
def build_advanced_bilstm(time_steps, n_features):
    model = Sequential([
        Bidirectional(LSTM(192, return_sequences=True), input_shape=(time_steps, n_features)),
        BatchNormalization(),
        Dropout(0.35),

        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.35),

        Bidirectional(LSTM(64)),
        Dropout(0.25),

        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# --------------------------------
# Find Best Threshold (F1-based)
# --------------------------------
def find_best_threshold(y_true, y_probs):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.2, 0.8, 0.01):
        y_pred = (y_probs > t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n✅ Best Threshold (F1): {best_t:.2f}")
    return best_t

# --------------------------------
# Train and evaluate
# --------------------------------
def train_and_evaluate(epochs=100):
    X_train, X_test, y_train, y_test = prepare_fall_detection_data(
        time_steps=60,
        step_size=2
    )
    if X_train is None:
        return

    # Class weights
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    # Model
    model = build_advanced_bilstm(X_train.shape[1], X_train.shape[2])
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_bilstm_advanced.keras", monitor='val_loss', save_best_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=6)

    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop, checkpoint, lr_reduce]
    )

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"\n✅ Training finished. Best epoch: {best_epoch}")

    # Evaluation
    loss, acc, prec, rec = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # Save final model
    model.save("bilstm_advanced_final.keras")
    print("\n✅ Final model saved")

    # Prediction
    y_probs = model.predict(X_test)
    threshold = find_best_threshold(y_test, y_probs)
    y_pred = (y_probs > threshold).astype(int)

    # Report
    print(f"\n--- Classification Report (Threshold={threshold:.2f}) ---")
    print(classification_report(y_test, y_pred, target_names=['Not Fall', 'Fall']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix (Threshold={threshold:.2f})")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate(epochs=100)
