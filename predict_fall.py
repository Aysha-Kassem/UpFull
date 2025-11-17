import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import data preparation function to get the test set
# Must be in the same directory as data_preprocessor.py
from data_preprocessor import prepare_fall_detection_data 

def predict_and_evaluate(model_path='fall_detection_model.h5'):
    """
    Loads the saved model, makes predictions on the test set, and reports results.
    It also plots the Confusion Matrix.
    """
    try:
        # Load the saved model
        model = load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not load the model from {model_path}. Error: {e}")
        print("Please ensure you have run 'train_model.py' successfully to save the model.")
        return

    # Prepare data again to get the X_test and y_test splits (using the same settings)
    _, X_test, _, y_test = prepare_fall_detection_data(
        filepath='DataSet.csv', 
        time_steps=50, 
        test_size=0.2, 
        random_state=42
    )

    if X_test is None:
        print("Error: Test data could not be prepared.")
        return

    print("\n--- Making Predictions on Test Data ---")
    
    # Get probability scores
    y_pred_probs = model.predict(X_test, verbose=1)
    
    # Convert probabilities to binary predictions (0 or 1)
    # Threshold is 0.5: if probability > 0.5, predict Fall (1), otherwise Not Fall (0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # --- Classification Report (detailed metrics) ---
    print("\n--- Classification Report ---")
    print("This shows detailed performance for both 'Not Fall' (0) and 'Fall' (1).")
    print(classification_report(y_test, y_pred, target_names=['Not Fall (0)', 'Fall (1)']))

    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, 
        annot=True, # Show the numbers in the squares
        fmt='d', # Format numbers as integers
        cmap='Blues', # Color scheme
        xticklabels=['Predicted 0 (Not Fall)', 'Predicted 1 (Fall)'], 
        yticklabels=['Actual 0 (Not Fall)', 'Actual 1 (Fall)']
    )
    plt.title('Confusion Matrix for Fall Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show() # Display the plot
    print("✅ Prediction and Visualization Complete.")


if __name__ == '__main__':
    predict_and_evaluate()