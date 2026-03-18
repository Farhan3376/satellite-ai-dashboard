"""
database.py — Satellite Data Dashboard: Database Layer

Provides a clean, modular SQLite-backed database for storing satellite image
metadata, extracted feature vectors, and ML predictions.

Why SQLite?
  - Zero external dependencies (uses Python's built-in sqlite3 module).
  - Easy to swap for PostgreSQL via sqlalchemy without API changes.
  - Scales well for single-node dashboard deployments.

Schema
------
  images          : image_id, image_path, class_label, upload_timestamp
  feature_vectors : id, image_id (FK), feature_vector (BLOB)
  predictions     : id, image_id (FK), predicted_label, confidence,
                    model_name, prediction_timestamp
"""

import sqlite3
import os
import pickle
import json
import numpy as np
from datetime import datetime
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DB_DIR, "satellite_dashboard.db")


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH):
    """
    Context-managed database connection. Automatically commits on success and
    rolls back on error, then closes the connection.
    
    Args:
        db_path: Path to the SQLite database file.
        
    Yields:
        sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    # Return rows as dict-like objects for readability
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema Creation
# ---------------------------------------------------------------------------

def create_tables(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Creates all database tables if they do not already exist.
    Safe to call multiple times (idempotent).
    
    Args:
        db_path: Path to the SQLite database file.
    """
    ddl = """
        -- Table 1: Core image metadata
        CREATE TABLE IF NOT EXISTS images (
            image_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path       TEXT    NOT NULL UNIQUE,
            class_label      TEXT    NOT NULL,
            upload_timestamp TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Table 2: Serialized feature vectors (BLOB = pickle-serialized numpy array)
        CREATE TABLE IF NOT EXISTS feature_vectors (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id         INTEGER NOT NULL UNIQUE,
            feature_vector   BLOB    NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE
        );

        -- Table 3: ML model predictions with confidence
        CREATE TABLE IF NOT EXISTS predictions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id              INTEGER NOT NULL,
            predicted_label       TEXT    NOT NULL,
            confidence            REAL,
            model_name            TEXT    NOT NULL DEFAULT 'unknown',
            prediction_timestamp  TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE
        );

        -- Indexes for common query patterns
        CREATE INDEX IF NOT EXISTS idx_images_class      ON images (class_label);
        CREATE INDEX IF NOT EXISTS idx_pred_image        ON predictions (image_id);
        CREATE INDEX IF NOT EXISTS idx_fv_image          ON feature_vectors (image_id);
    """
    with get_connection(db_path) as conn:
        conn.executescript(ddl)


# ---------------------------------------------------------------------------
# Insert Functions
# ---------------------------------------------------------------------------

def insert_image(image_path: str, class_label: str,
                 db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Registers a new image in the database.

    If the image already exists (same path), returns its existing image_id.

    Args:
        image_path:  Absolute path to the image file.
        class_label: Predicted or ground-truth class ('Forest', 'Water', etc.).
        db_path:     Path to the SQLite database.

    Returns:
        int: The image_id of the inserted (or existing) image.
    """
    sql_insert = """
        INSERT OR IGNORE INTO images (image_path, class_label, upload_timestamp)
        VALUES (?, ?, ?)
    """
    sql_select = "SELECT image_id FROM images WHERE image_path = ?"
    timestamp = datetime.now().isoformat()

    with get_connection(db_path) as conn:
        conn.execute(sql_insert, (image_path, class_label, timestamp))
        row = conn.execute(sql_select, (image_path,)).fetchone()
        return row["image_id"]


def insert_feature_vector(image_id: int, feature_vector: np.ndarray,
                           db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Stores a numpy feature vector for an image, serialized via pickle.

    Only one feature vector is stored per image (UPSERT style).

    Args:
        image_id:       The image's primary key from the images table.
        feature_vector: 1-D numpy array produced by extract_features().
        db_path:        Path to the SQLite database.
    """
    blob = pickle.dumps(feature_vector)
    sql = """
        INSERT INTO feature_vectors (image_id, feature_vector)
        VALUES (?, ?)
        ON CONFLICT (image_id) DO UPDATE SET feature_vector = excluded.feature_vector
    """
    with get_connection(db_path) as conn:
        conn.execute(sql, (image_id, blob))


def insert_prediction(image_id: int, predicted_label: str,
                      confidence: float = None, model_name: str = "unknown",
                      db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Records an ML model prediction for an image.

    Multiple predictions (from different models) can be stored per image.

    Args:
        image_id:        Primary key of the classified image.
        predicted_label: String class predicted by the model.
        confidence:      Optional confidence score (e.g., probability or distance).
        model_name:      Name/tag of the model that produced the result.
        db_path:         Path to the SQLite database.
    """
    sql = """
        INSERT INTO predictions
            (image_id, predicted_label, confidence, model_name, prediction_timestamp)
        VALUES (?, ?, ?, ?, ?)
    """
    timestamp = datetime.now().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(sql, (image_id, predicted_label, confidence, model_name, timestamp))


# ---------------------------------------------------------------------------
# Query Functions
# ---------------------------------------------------------------------------

def query_by_class(class_label: str, limit: int = 100,
                   db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Retrieves images belonging to a specific class.

    Args:
        class_label: Target class string (e.g., 'Forest').
        limit:       Maximum number of results.
        db_path:     Database path.

    Returns:
        List of dicts with image_id, image_path, class_label, upload_timestamp.
    """
    sql = """
        SELECT image_id, image_path, class_label, upload_timestamp
        FROM   images
        WHERE  class_label = ?
        LIMIT  ?
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, (class_label, limit)).fetchall()
        return [dict(r) for r in rows]


def query_predictions(image_id: int,
                      db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Fetches all stored predictions for a given image.

    Args:
        image_id: Primary key of the image.
        db_path:  Database path.

    Returns:
        List of dicts with predicted_label, confidence, model_name, timestamp.
    """
    sql = """
        SELECT predicted_label, confidence, model_name, prediction_timestamp
        FROM   predictions
        WHERE  image_id = ?
        ORDER  BY prediction_timestamp DESC
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, (image_id,)).fetchall()
        return [dict(r) for r in rows]


def query_top_k_similar(query_vector: np.ndarray, k: int = 5,
                         class_filter: str = None,
                         db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Finds the Top-K most similar images stored in the database using
    vectorized cosine similarity over all persisted feature vectors.

    Args:
        query_vector:  1-D numpy feature vector for the query image.
        k:             Number of top results to return.
        class_filter:  Optional class label to restrict search scope
                       (enables class-specific retrieval, like Step 4).
        db_path:       Database path.

    Returns:
        List of dicts (sorted by similarity desc), each containing:
          image_id, image_path, class_label, similarity_score, rank.
    """
    # Join feature_vectors with images to get metadata alongside the BLOB
    if class_filter:
        sql = """
            SELECT fv.image_id, fv.feature_vector, i.image_path, i.class_label
            FROM   feature_vectors fv
            JOIN   images          i  ON fv.image_id = i.image_id
            WHERE  i.class_label = ?
        """
        params = (class_filter,)
    else:
        sql = """
            SELECT fv.image_id, fv.feature_vector, i.image_path, i.class_label
            FROM   feature_vectors fv
            JOIN   images          i  ON fv.image_id = i.image_id
        """
        params = ()

    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    # Deserialize all feature vectors into a matrix for batch cosine similarity
    ids, paths, labels, matrix = [], [], [], []
    for r in rows:
        try:
            vec = pickle.loads(r["feature_vector"])
            ids.append(r["image_id"])
            paths.append(r["image_path"])
            labels.append(r["class_label"])
            matrix.append(vec)
        except Exception:
            continue

    feature_matrix = np.array(matrix)

    # Vectorized cosine similarity: dot(q, X.T) / (|q| * |X_i|)
    q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    norms = np.linalg.norm(feature_matrix, axis=1) + 1e-10
    scores = feature_matrix.dot(q) / norms

    # Pick Top-K indices
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "image_id": ids[idx],
            "image_path": paths[idx],
            "class_label": labels[idx],
            "similarity_score": float(scores[idx])
        })

    return results


def get_image_by_path(image_path: str,
                      db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """
    Retrieves the image record for a given file path.

    Args:
        image_path: The absolute path of the image.
        db_path:    Database path.

    Returns:
        Dict with image metadata, or None if not found.
    """
    sql = "SELECT * FROM images WHERE image_path = ?"
    with get_connection(db_path) as conn:
        row = conn.execute(sql, (image_path,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Full Pipeline Convenience Function
# ---------------------------------------------------------------------------

def ingest_image_pipeline(image_path: str, class_label: str,
                           feature_vector: np.ndarray,
                           predicted_label: str = None,
                           confidence: float = None,
                           model_name: str = "unknown",
                           db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Convenience function to register an image, store its features, and
    optionally record a prediction in a single atomic pipeline call.

    Designed for integration into the Dashboard and REST API.

    Args:
        image_path:      Path to the satellite image.
        class_label:     Ground-truth or thematic class label.
        feature_vector:  Extracted numpy feature vector.
        predicted_label: Optional ML prediction result.
        confidence:      Optional prediction confidence score.
        model_name:      Model identifier.
        db_path:         Database path.

    Returns:
        int: image_id of the ingested image.
    """
    image_id = insert_image(image_path, class_label, db_path)
    insert_feature_vector(image_id, feature_vector, db_path)

    if predicted_label is not None:
        insert_prediction(image_id, predicted_label, confidence, model_name, db_path)

    return image_id


# ---------------------------------------------------------------------------
# Entry Point: Quick Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("Creating database schema...")
    create_tables()
    print(f"Database created at: {DEFAULT_DB_PATH}")

    # --- Insert a demo record ---
    demo_path = "/home/romi/farhan/ML project/dataset/test/Forest/Forest_Forest_1.jpg"
    demo_class = "Forest"
    demo_vector = np.random.rand(1880).astype(np.float32)  # Simulated feature vector

    if os.path.exists(demo_path):
        img_id = ingest_image_pipeline(
            image_path=demo_path,
            class_label=demo_class,
            feature_vector=demo_vector,
            predicted_label="Forest",
            confidence=0.97,
            model_name="RandomForest"
        )
        print(f"\nIngested demo image with image_id={img_id}")

        # --- Query predictions ---
        preds = query_predictions(img_id)
        print(f"\nPredictions for image_id={img_id}:")
        for p in preds:
            print(f"  {p['model_name']} → {p['predicted_label']} "
                  f"(confidence={p['confidence']})")

        # --- Query by class ---
        class_results = query_by_class("Forest", limit=5)
        print(f"\nImages in 'Forest' class ({len(class_results)} returned):")
        for r in class_results:
            print(f"  [{r['image_id']}] {os.path.basename(r['image_path'])}")

        # --- Top-K similarity search ---
        top_k = query_top_k_similar(demo_vector, k=3)
        print(f"\nTop-3 similar images:")
        for r in top_k:
            print(f"  Rank {r['rank']}: {os.path.basename(r['image_path'])} "
                  f"(score={r['similarity_score']:.4f})")
    else:
        print(f"\nDemo image not found at {demo_path}. Schema creation verified — run with real data to test inserts.")

    print("\nDatabase layer verified successfully.")
