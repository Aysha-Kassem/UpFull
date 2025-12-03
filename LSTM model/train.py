"""
Train LSTM for fall detection using DataSet.csv
- Uses prepare_fall_detection_data from data_preprocessor.py
- Plots training curves and confusion matrix
- Saves final model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessor import prepare_fall_detection_data

def build_lstm_model(time_steps, n_features, dropout_rate=0.3):
    model = Sequential()
    model.add(LSTM(128, input_shape=(time_steps, n_features), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(64))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

def train_model():
    # 1) load & preprocess data
    X_train, X_test, y_train, y_test = prepare_fall_detection_data(
        filepath='DataSet.csv', time_steps=50, step_size=2
    )
    
    if X_train is None:
        print("❌ Preprocessing failed")
        return

    print("Shapes:", X_train.shape, y_train.shape)

    # 2) build model
    model = build_lstm_model(time_steps=X_train.shape[1], n_features=X_train.shape[2])
    model.summary()

    # 3) callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1)
    checkpoint = ModelCheckpoint('best_fall_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

    # 4) train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=80,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # 5) evaluation
    print("\n✅ Training finished.")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")

    # predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Not Fall','Fall']))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # plot training curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.tight_layout()
    plt.show()

    # save final model
    model.save('final_fall_model.keras')
    print("✅ Final model saved as final_fall_model.keras")

if __name__ == "__main__":
    train_model()
