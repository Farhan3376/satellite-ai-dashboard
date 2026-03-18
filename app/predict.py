import os
import joblib
try:
    from app.extract_features import extract_features
except ImportError:
    from extract_features import extract_features

def predict_image(image_path, model_path, scaler_path, encoder_path):
    """
    Predicts the class of a single satellite image using a trained model.
    Runs the full pipeline seamlessly: extracting features, normalizing them, and running inference.
    
    Args:
        image_path (str): Path to the image to classify.
        model_path (str): Path to the trained .joblib model.
        scaler_path (str): Path to the fitted StandardScaler .joblib from training.
        encoder_path (str): Path to the fitted LabelEncoder .joblib from training.
        
    Returns:
        str: The predicted string class for the image.
    """
    # 1. Extract raw features reusing our step 2 logic
    features = extract_features(image_path)
    if features is None:
        raise ValueError(f"Could not load or extract features from image: {image_path}")
        
    # Reshape features to a 2D array for scikit-learn (1 sample, n_features)
    features_2d = features.reshape(1, -1)
    
    # 2. Load the StandardScaler and normalize the features
    scaler = joblib.load(scaler_path)
    features_normalized = scaler.transform(features_2d)
    
    # 3. Load the model and execute the prediction
    model = joblib.load(model_path)
    prediction_encoded = model.predict(features_normalized)
    
    # 4. Decode the integer label to its original string mapping
    encoder = joblib.load(encoder_path)
    predicted_class = encoder.inverse_transform(prediction_encoded)[0]
    
    return predicted_class

if __name__ == "__main__":
    # Define environment paths required for the predictive pipeline
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(APP_DIR)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Using the Random Forest model for this test execution
    MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
    ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")
    
    # Testing the inference cleanly with a sample test image
    SAMPLE_IMAGE = os.path.join(BASE_DIR, "dataset", "test", "Forest", "Forest_Forest_5.jpg")
    
    if os.path.exists(SAMPLE_IMAGE):
        print(f"Running prediction pipeline on: {SAMPLE_IMAGE}")
        predicted_class = predict_image(SAMPLE_IMAGE, MODEL_PATH, SCALER_PATH, ENCODER_PATH)
        print(f"\n=> The model predicts this image is: {predicted_class}")
    else:
        print(f"Sample image {SAMPLE_IMAGE} not found.") 
        print("Please provide a valid path to predict_image() or adjust SAMPLE_IMAGE in predict.py.")
