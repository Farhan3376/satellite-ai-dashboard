import os
import joblib
try:
    from app.extract_features import extract_features
except ImportError:
    from extract_features import extract_features

def predict_image(image_path, model_path, scaler_path, encoder_path, pca_path=None):
    """
    Predicts the class of a single satellite image using a trained model.
    Runs the full pipeline seamlessly: extracting features, normalizing them, 
    applying PCA (optional), and running inference.
    """
    # 1. Extract raw features
    features = extract_features(image_path)
    if features is None:
        raise ValueError(f"Could not load or extract features from image: {image_path}")
        
    features_2d = features.reshape(1, -1)
    
    # 2. Normalize features
    scaler = joblib.load(scaler_path)
    features_normalized = scaler.transform(features_2d)
    
    # 3. Apply PCA if provided
    if pca_path and os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        features_normalized = pca.transform(features_normalized)
    
    # 4. Model Prediction
    model = joblib.load(model_path)
    prediction_encoded = model.predict(features_normalized)
    
    # 5. Decode label
    encoder = joblib.load(encoder_path)
    predicted_class = encoder.inverse_transform(prediction_encoded)[0]
    
    return predicted_class

if __name__ == "__main__":
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(APP_DIR)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
    ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")
    PCA_PATH = os.path.join(MODELS_DIR, "pca_transformer.joblib")
    
    SAMPLE_IMAGE = os.path.join(BASE_DIR, "dataset", "test", "Forest", "Forest_Forest_5.jpg")
    
    if os.path.exists(SAMPLE_IMAGE):
        print(f"Running prediction pipeline on: {SAMPLE_IMAGE}")
        predicted_class = predict_image(SAMPLE_IMAGE, MODEL_PATH, SCALER_PATH, ENCODER_PATH, PCA_PATH)
        print(f"\n=> The model predicts this image is: {predicted_class}")
    else:
        print(f"Sample image {SAMPLE_IMAGE} not found.") 
        print("Please provide a valid path to predict_image() or adjust SAMPLE_IMAGE in predict.py.")
