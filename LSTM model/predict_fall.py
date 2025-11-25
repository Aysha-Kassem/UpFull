import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessor import prepare_fall_detection_data


def predict_and_evaluate(model_path='best_bilstm_fall_detection_model.keras', threshold=0.42):
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # Load test data
    _, X_test, _, y_test = prepare_fall_detection_data(
        time_steps=50,
        step_size=5
    )

    if X_test is None:
        return

    # Predict
    y_probs = model.predict(X_test)
    y_pred = (y_probs > threshold).astype(int)

    # Ensure correct shapes
    y_test_flat = y_test.ravel()
    y_pred_flat = y_pred.ravel()

    # Classification report
    print(f"\n--- Classification Report (Threshold={threshold}) ---")
    print(classification_report(y_test_flat, y_pred_flat, target_names=["Not Fall", "Fall"]))

    # Confusion matrix
    cm = confusion_matrix(y_test_flat, y_pred_flat)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Calculate Error Rate correctly ✅
    total_samples = len(y_test_flat)
    correct_predictions = (y_pred_flat == y_test_flat).sum()
    incorrect_predictions = (y_pred_flat != y_test_flat).sum()

    accuracy = correct_predictions / total_samples
    error_rate = incorrect_predictions / total_samples

    print("\n--- Additional Metrics ---")
    print(f"✅ Total Samples: {total_samples}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"❌ Incorrect Predictions: {incorrect_predictions}")
    print(f"📌 Accuracy: {accuracy * 100:.2f}%")
    print(f"📌 Error Rate: {error_rate * 100:.2f}%")


if __name__ == "__main__":
    predict_and_evaluate(
        model_path="best_bilstm_fall_detection_model.keras",
        threshold=0.42
    )

