# ============================================
# train_best_model.py
# ============================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessor import prepare_early_fall_data

# ============================================================
# 🟪 أفضل موديل فعلي Multi-Head BiLSTM + Attention
# ============================================================

def attention_block(inputs):
    score = Dense(inputs.shape[-1], activation='tanh')(inputs)
    score = Dense(1)(score)
    weights = tf.nn.softmax(score, axis=1)
    context = tf.reduce_sum(weights * inputs, axis=1)
    return context

def build_best_bilstm(time_steps, n_features, dropout=0.35):

    inp = Input(shape=(time_steps, n_features))

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    # 🟦 Attention
    att = attention_block(x)

    # fall_now
    h_now = Dense(64, activation='relu')(att)
    h_now = Dropout(0.25)(h_now)
    out_now = Dense(1, activation='sigmoid', name='fall_now')(h_now)

    # fall_soon
    h_soon = Dense(64, activation='relu')(att)
    h_soon = Dropout(0.25)(h_soon)
    out_soon = Dense(1, activation='sigmoid', name='fall_soon')(h_soon)

    model = Model(inp, [out_now, out_soon])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={'fall_now': 'binary_crossentropy', 'fall_soon': 'binary_crossentropy'},
        loss_weights={'fall_now': 1.0, 'fall_soon': 1.3},
        metrics={
            'fall_now': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            'fall_soon': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )

    return model

# ============================================================
# Training + Evaluation + Plots
# ============================================================

def train_and_evaluate(time_steps=50, epochs=70, batch_size=32, step_size=2):

    X_train, X_test, y_now_train, y_now_test, y_soon_train, y_soon_test = \
        prepare_early_fall_data(time_steps=time_steps, step_size=step_size)

    model = build_best_bilstm(time_steps, X_train.shape[2])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5)

    history = model.fit(
        X_train,
        {"fall_now": y_now_train, "fall_soon": y_soon_train},
        validation_data=(X_test, {"fall_now": y_now_test, "fall_soon": y_soon_test}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, lr_reduce],
        verbose=1
    )

    # ======================================================
    # 📊 رسم الجرافات كاملة
    # ======================================================

    # ---- LOSS ----
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss Curve"); plt.grid()
    plt.savefig("loss.png"); plt.show()

    # ---- Accuracy ----
    plt.figure(figsize=(6,4))
    plt.plot(history.history["fall_now_accuracy"], label="Fall Now Acc")
    plt.plot(history.history["val_fall_now_accuracy"], label="Val Fall Now Acc")
    plt.plot(history.history["fall_soon_accuracy"], label="Fall Soon Acc")
    plt.plot(history.history["val_fall_soon_accuracy"], label="Val Fall Soon Acc")
    plt.legend(); plt.title("Accuracy Curve"); plt.grid()
    plt.savefig("accuracy.png"); plt.show()

    # ---- Precision ----
    plt.figure(figsize=(6,4))
    plt.plot(history.history["fall_now_precision"], label="Fall Now Prec")
    plt.plot(history.history["val_fall_now_precision"], label="Val Fall Now Prec")
    plt.plot(history.history["fall_soon_precision"], label="Fall Soon Prec")
    plt.plot(history.history["val_fall_soon_precision"], label="Val Fall Soon Prec")
    plt.legend(); plt.title("Precision Curve"); plt.grid()
    plt.savefig("precision.png"); plt.show()

    # ---- Recall ----
    plt.figure(figsize=(6,4))
    plt.plot(history.history["fall_now_recall"], label="Fall Now Recall")
    plt.plot(history.history["val_fall_now_recall"], label="Val Fall Now Recall")
    plt.plot(history.history["fall_soon_recall"], label="Fall Soon Recall")
    plt.plot(history.history["val_fall_soon_recall"], label="Val Fall Soon Recall")
    plt.legend(); plt.title("Recall Curve"); plt.grid()
    plt.savefig("recall.png"); plt.show()

    # ======================================================
    # 📘 Predictions
    # ======================================================

    prob_now, prob_soon = model.predict(X_test)
    prob_now = prob_now.flatten()
    prob_soon = prob_soon.flatten()

    pred_now = (prob_now >= 0.5).astype(int)
    pred_soon = (prob_soon >= 0.45).astype(int)

    # ======================================================
    # Confusion Matrices
    # ======================================================

    cm_now = confusion_matrix(y_now_test, pred_now)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_now, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Fall NOW")
    plt.savefig("cm_fall_now.png")
    plt.show()

    cm_soon = confusion_matrix(y_soon_test, pred_soon)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_soon, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Fall SOON")
    plt.savefig("cm_fall_soon.png")
    plt.show()

    # ======================================================
    # ROC Curve (Fall Soon)
    # ======================================================

    fpr, tpr, _ = roc_curve(y_soon_test, prob_soon)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    plt.plot([0,1], [0,1], "k--")
    plt.legend(); plt.grid()
    plt.title("ROC Curve — Early Fall Prediction")
    plt.savefig("roc_fall_soon.png")
    plt.show()

    # Save model
    model.save("final_best_model.keras")
    print("✅ Saved final_best_model.keras")


if __name__ == "__main__":
    train_and_evaluate()
