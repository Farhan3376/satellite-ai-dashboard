import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(features_dir):
    """Loads normalized X and encoded y from disk."""
    X_train = np.load(os.path.join(features_dir, "X_train.npy"))
    y_train = np.load(os.path.join(features_dir, "y_train.npy"))
    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))
    
    # Load encoder to get class names for the classification report
    encoder = joblib.load(os.path.join(features_dir, "encoder.joblib"))
    target_names = encoder.classes_
    
    return X_train, y_train, X_test, y_test, target_names

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, target_names, models_dir):
    print("\n--- Training SVM (RBF Kernel) ---")
    
    # Initialize SVM with RBF kernel
    # cache_size is increased to 2000MB to accelerate training on large datasets
    svm_model = SVC(kernel='rbf', cache_size=2000, random_state=42)
    
    # Fit the model
    svm_model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {acc:.4f}")
    print("\nSVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the trained model to disk
    model_path = os.path.join(models_dir, "svm_model.joblib")
    joblib.dump(svm_model, model_path)
    print(f"Saved SVM model to {model_path}")
    
    return svm_model

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, target_names, models_dir):
    print("\n--- Training Random Forest Classifier ---")
    
    # Initialize Random Forest, using all available cores (n_jobs=-1)
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    # Fit the model
    rf_model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {acc:.4f}")
    print("\nRandom Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the trained model to disk
    model_path = os.path.join(models_dir, "rf_model.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Saved Random Forest model to {model_path}")
    
    return rf_model

if __name__ == "__main__":
    FEATURES_DIR = "/home/romi/farhan/ML project/features_output"
    MODELS_DIR = "/home/romi/farhan/ML project/models"
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Loading data from {FEATURES_DIR} ...")
    X_train, y_train, X_test, y_test, target_names = load_data(FEATURES_DIR)
    
    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")
    print(f"Feature vector dimensionality: {X_train.shape[1]}")
    
    # 1. Train and evaluate Support Vector Machine
    train_and_evaluate_svm(X_train, y_train, X_test, y_test, target_names, MODELS_DIR)
    
    # 2. Train and evaluate Random Forest Classifier
    train_and_evaluate_rf(X_train, y_train, X_test, y_test, target_names, MODELS_DIR)
    
    print("\nModel training and evaluation successfully completed.")
