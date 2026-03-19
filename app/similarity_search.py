import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

try:
    from app.extract_features import extract_features
except ImportError:
    from extract_features import extract_features

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
def load_feature_index(features_dir, dataset_path):
    """
    Loads the precomputed feature matrix consolidated for search.
    
    Args:
        features_dir (str): Path to the folder containing model files.
        dataset_path (str): Path to the structured dataset root folder.
        
    Returns:
        tuple: (all_features: np.ndarray, all_labels: np.ndarray, all_paths: list)
    """
    # 1. Load the binary features, labels, and relative paths
    X_path = os.path.join(features_dir, "X_train_v2.npy")
    y_path = os.path.join(features_dir, "y_train_v2.npy")
    p_path = os.path.join(features_dir, "paths_v2.npy")
    
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    relative_paths = np.load(p_path, allow_pickle=True)
    
    # 2. Rebuild the absolute image paths for the current environment
    all_paths = [os.path.join(dataset_path, p) for p in relative_paths]
    
    return X, y, all_paths
    
    return X, y, all_paths

# ---------------------------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------------------------

def compute_similarity(query_vec, feature_matrix, metric='cosine'):
    """
    Computes similarity between a query vector and all vectors in the feature matrix
    using vectorized numpy/sklearn operations for large-scale efficiency.

    Args:
        query_vec (np.ndarray): The 1D feature vector of the query image.
        feature_matrix (np.ndarray): The (N, D) feature matrix for the full dataset.
        metric (str): Similarity metric: 'cosine' (higher=closer) or 'euclidean' (lower=closer).

    Returns:
        np.ndarray: 1D array of similarity scores (all values, same length as feature_matrix).
    """
    query_2d = query_vec.reshape(1, -1)

    if metric == 'cosine':
        # Returns values in [-1, 1]; higher means more similar
        scores = cosine_similarity(query_2d, feature_matrix)[0]
    elif metric == 'euclidean':
        # Returns distance; lower means more similar — we negate for consistent ordering
        scores = -euclidean_distances(query_2d, feature_matrix)[0]
    else:
        raise ValueError(f"Unsupported metric: '{metric}'. Choose 'cosine' or 'euclidean'.")

    return scores

def find_top_k_similar(query_image_path, feature_matrix, all_paths, scaler,
                        k=5, metric='cosine', pca_transformer=None):
    """
    Returns the Top-K most similar images from the indexed dataset for a given query image.

    Args:
        query_image_path (str): Path to the query image.
        feature_matrix (np.ndarray): Preloaded (N, D) feature matrix for all indexed images.
        all_paths (list): List of image paths corresponding to rows in feature_matrix.
        scaler: Fitted StandardScaler to normalize the extracted query features.
        k (int): Number of top similar images to return.
        metric (str): 'cosine' or 'euclidean'.
        pca_transformer: Optional fitted PCA model to reduce query dimensions.

    Returns:
        list[dict]: Top-K results, each with 'path', 'score', and 'rank'.
    """
    # 1. Extract raw features from the query image
    raw_features = extract_features(query_image_path)
    if raw_features is None:
        raise ValueError(f"Could not extract features from: {query_image_path}")

    # 2. Normalize the query features using the training scaler to match the indexed scale
    query_normalized = scaler.transform(raw_features.reshape(1, -1))

    # 3. Apply PCA transformation if searching against a reduced index
    if pca_transformer:
        query_normalized = pca_transformer.transform(query_normalized)
    
    query_normalized = query_normalized[0]

    # 4. Compute similarity scores against all indexed images (fully vectorized)
    scores = compute_similarity(query_normalized, feature_matrix, metric=metric)

    # 4. Find the Top-K indices using argpartition for efficiency on large datasets
    top_k_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_k_indices, start=1):
        results.append({
            'rank': rank,
            'path': all_paths[idx],
            'score': float(scores[idx])
        })

    return results

# ---------------------------------------------------------------------------
# Optional Visualization
# ---------------------------------------------------------------------------

def visualize_top_k(query_image_path, results, output_path=None):
    """
    Renders a side-by-side grid of the query image and the top-K similar results.

    Args:
        query_image_path (str): Path to the query image.
        results (list[dict]): Output from find_top_k_similar().
        output_path (str, optional): If provided, saves the figure to this path.
    """
    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))
    fig.suptitle("Top-K Similar Image Retrieval", fontsize=14, fontweight='bold')

    # Display the query image in the first column
    query_img = cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=10, fontweight='bold', color='blue')
    axes[0].axis('off')

    # Display each retrieved image with its rank and score
    for i, result in enumerate(results):
        img = cv2.imread(result['path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = os.path.basename(os.path.dirname(result['path']))
        axes[i + 1].imshow(img_rgb)
        axes[i + 1].set_title(
            f"#{result['rank']} | {label}\nScore: {result['score']:.4f}",
            fontsize=8
        )
        axes[i + 1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import joblib

    FEATURES_DIR = "/home/romi/farhan/ML project/features_output"
    DATASET_PATH = "/home/romi/farhan/ML project/dataset"
    SCALER_PATH = os.path.join(FEATURES_DIR, "scaler.joblib")
    OUTPUT_VIZ = "/home/romi/farhan/ML project/similarity_results.png"

    # Load the fitted scaler from Step 2
    scaler = joblib.load(SCALER_PATH)

    # Build the indexed feature database from all splits
    print("Building feature index from all splits...")
    feature_matrix, label_array, all_paths = load_feature_index(FEATURES_DIR, DATASET_PATH)
    print(f"Feature index built: {feature_matrix.shape[0]} images indexed.")

    # Select a sample query image from the test split
    QUERY_IMAGE = os.path.join(DATASET_PATH, "test", "Forest", "Forest_Forest_9.jpg")

    if not os.path.exists(QUERY_IMAGE):
        # Fallback to first available image in test split
        QUERY_IMAGE = all_paths[-1]

    print(f"\nQuery image: {QUERY_IMAGE}")
    print("Searching for Top-5 similar images using cosine similarity...")

    # Perform the similarity search
    results = find_top_k_similar(
        query_image_path=QUERY_IMAGE,
        feature_matrix=feature_matrix,
        all_paths=all_paths,
        scaler=scaler,
        k=5,
        metric='cosine'
    )

    print("\n--- Top-5 Similar Images ---")
    for r in results:
        label = os.path.basename(os.path.dirname(r['path']))
        print(f"  Rank {r['rank']}: [{label}] {os.path.basename(r['path'])}  |  Score: {r['score']:.6f}")

    # Optionally visualize and save the results
    print("\nGenerating visualization grid...")
    visualize_top_k(QUERY_IMAGE, results, output_path=OUTPUT_VIZ)
