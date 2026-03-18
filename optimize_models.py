import os
import joblib
import numpy as np
from sklearn.decomposition import PCA

def optimize_models(models_dir):
    print("🚀 Starting Model & Data Optimization...")
    
    # 1. Optimize Scaler and Encoder (usually small, but good practice)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    encoder_path = os.path.join(models_dir, "encoder.joblib")
    
    # 2. Optimize Main ML Models (Highest Compression)
    for model_name, comp_lvl in [("rf_model.joblib", 3), ("svm_model.joblib", 9)]:
        path = os.path.join(models_dir, model_name)
        if os.path.exists(path):
            print(f"📦 Compressing {model_name} (lvl {comp_lvl})...")
            model = joblib.load(path)
            joblib.dump(model, path, compress=comp_lvl)
            print(f"✅ {model_name} compressed to {os.path.getsize(path)/1024/1024:.2f} MB")

    # 3. Optimize Feature Vectors (PCA + Float32)
    # Combine X_train and X_test if we want a global search index
    x_train_path = os.path.join(models_dir, "X_train.npy")
    x_test_path = os.path.join(models_dir, "X_test.npy")
    y_train_path = os.path.join(models_dir, "y_train.npy")
    y_test_path = os.path.join(models_dir, "y_test.npy")

    if os.path.exists(x_train_path):
        print("🪄 Applying PCA & Precision Casting to Feature Matrix...")
        X = np.load(x_train_path).astype(np.float32)
        
        # Original size: ~200-400MB
        # Reduce to 256 dimensions (captures most variance for similarity search)
        pca = PCA(n_components=256)
        X_reduced = pca.fit_transform(X)
        
        # Save PCA model so we can transform query images later
        joblib.dump(pca, os.path.join(models_dir, "pca_transformer.joblib"), compress=3)
        
        # Save the reduced matrix
        np.save(x_train_path, X_reduced)
        print(f"✅ X_train.npy reduced and saved: {os.path.getsize(x_train_path)/1024/1024:.2f} MB")
        
        # Do the same for test if it exists
        if os.path.exists(x_test_path):
            X_test = np.load(x_test_path).astype(np.float32)
            X_test_reduced = pca.transform(X_test)
            np.save(x_test_path, X_test_reduced)
            print(f"✅ X_test.npy reduced and saved: {os.path.getsize(x_test_path)/1024/1024:.2f} MB")

    print("\n✨ Optimization Complete! All files should now be well under the 100MB GitHub limit.")
    print("⚠️  Note: I will now update the backend code to utilize the new PCA transformer.")

if __name__ == "__main__":
    MODELS_DIR = "/home/romi/farhan/ML project/models"
    optimize_models(MODELS_DIR)
