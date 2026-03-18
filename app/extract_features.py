import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def extract_features(image_path):
    """
    Extracts Color Histogram, GLCM texture, and HOG features from an image.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.array: A 1D numpy array comprising all concatenated features.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Color Histogram (RGB)
    hist_features = []
    # Loop over 3 channels (B, G, R)
    for i in range(3):
        # Calculate histogram with 32 bins per channel to reduce dimensionality
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        hist_features.extend(hist.flatten())
    
    hist_features = np.array(hist_features)
    
    # 2. GLCM Texture
    # Convert image to grayscale for texture and shape extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM
    # Evaluate at distance 1 across 4 angles (0, 45, 90, 135 degrees)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    
    # Extract structural properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    glcm_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
    
    # 3. HOG (Histogram of Oriented Gradients)
    # Extract shape attributes representing edges heavily utilized in satellite imagery
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # Concatenate all features into a single massive vector
    return np.hstack([hist_features, glcm_features, hog_features])

def load_and_process_dataset(dataset_path):
    """
    Traverses the dataset path, extracts features for all images, and returns train/test splits.
    Reuses the structure (train, val, test) provided in the dataset_path.
    
    Args:
        dataset_path (str): Path to the root dataset.
        
    Returns:
        dict: A dictionary containing features (X), labels (y), and original paths for train/val/test splits.
    """
    data = {}
    
    # Process train, val, and test data ensuring previous split is reused
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(split_dir):
            continue
            
        features = []
        labels = []
        paths = []
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Loading {split} split, class: {class_name}")
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, filename)
                    feat = extract_features(img_path)
                    
                    if feat is not None:
                        features.append(feat)
                        labels.append(class_name)
                        paths.append(img_path)
                        
        data[split] = {
            'X': np.array(features),
            'y': np.array(labels),
            'paths': paths
        }
        
    return data

def preprocess_features(data):
    """
    Normalizes features (zero mean, unit variance) and encodes class labels.
    
    Args:
        data (dict): Dictionary with train/val/test splits containing 'X' and 'y'.
        
    Returns:
        tuple: (processed_data, LabelEncoder, StandardScaler)
    """
    # Initialize objects required across ML pipeline
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    # We must fit the scaler and encoder on the TRAINING split only to avoid data leakage
    if 'train' not in data:
        raise ValueError("Training split is missing from the dataset.")
        
    X_train_raw = data['train']['X']
    y_train_raw = data['train']['y']
    
    print("Normalizing features and encoding labels...")
    
    # Fit & transform the training set
    X_train_norm = scaler.fit_transform(X_train_raw)
    y_train_encoded = encoder.fit_transform(y_train_raw)
    
    processed_data = {
        'train': (X_train_norm, y_train_encoded)
    }
    
    # Transform test and val sets if available, using the parameters fitted exclusively on Train Data
    for split in ['val', 'test']:
        if split in data:
            X_norm = scaler.transform(data[split]['X'])
            y_encoded = encoder.transform(data[split]['y'])
            processed_data[split] = (X_norm, y_encoded)
            
    return processed_data, encoder, scaler

if __name__ == "__main__":
    # Base dataset path assumed to be constructed by step 1
    BASE_DATASET_PATH = "/home/romi/farhan/ML project/dataset"
    OUTPUT_FEATURES_DIR = "/home/romi/farhan/ML project/features_output"
    
    print(f"Reading dataset from: {BASE_DATASET_PATH}")
    
    # 1. Load data and extract un-normalized features across image splits
    raw_data = load_and_process_dataset(BASE_DATASET_PATH)
    
    # 2. Normalize features & encode labels mathematically
    processed_data, label_encoder, feature_scaler = preprocess_features(raw_data)
    
    # Train and test now exist with 0 mean and 1 variance suitable for ML
    X_train, y_train = processed_data['train']
    X_test, y_test = processed_data.get('test', (None, None))
    
    print()
    print("--- Feature Extraction Summary ---")
    print(f"Classes encoded: {list(label_encoder.classes_)}")
    print(f"Feature vector dimensionality: {X_train.shape[1]}")
    print(f"Training split samples: {X_train.shape[0]}")
    if X_test is not None:
        print(f"Testing split samples:  {X_test.shape[0]}")
    
    # 3. Output features (X) and labels (Y) arrays specifically tailored for the Dashboard Model
    print(f"\nSaving preprocessed components to {OUTPUT_FEATURES_DIR} ...")
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_FEATURES_DIR, "y_train.npy"), y_train)
    
    if X_test is not None:
        np.save(os.path.join(OUTPUT_FEATURES_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(OUTPUT_FEATURES_DIR, "y_test.npy"), y_test)
        
    print("Saving scaler and encoder objects for dashboard deployment...")
    joblib.dump(feature_scaler, os.path.join(OUTPUT_FEATURES_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_FEATURES_DIR, "encoder.joblib"))
        
    print("Feature extraction complete. Preprocessed features are ready for ML integration.")
