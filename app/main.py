import os
import sys
import tempfile
import shutil
import logging
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import existing modules using package paths
try:
    from app.extract_features import extract_features
    from app.similarity_search import load_feature_index, find_top_k_similar
    from database.database import (
        create_tables, ingest_image_pipeline, get_connection, get_image_by_path
    )
except ImportError:
    # Local development fallback
    sys.path.append(os.path.join(ROOT_DIR, "app"))
    from extract_features import extract_features
    from similarity_search import load_feature_index, find_top_k_similar
    from database.database import (
        create_tables, ingest_image_pipeline, get_connection, get_image_by_path
    )

# ---------------------------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Satellite Intelligence API",
    description="REST API for Satellite Imagery Classification, Similarity Search, and Analytics.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants base on project root
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
UPLOADS_DIR = os.path.join(ROOT_DIR, "uploads")

# Use global variables to hold models and features to avoid reloading
SCALER = None
ENCODER = None
RF_MODEL = None
SVM_MODEL = None
PCA_TRANSFORMER = None
FEATURE_MATRIX = None
LABEL_ARRAY = None
ALL_PATHS = None

@app.on_event("startup")
def startup_event():
    """Load heavy modules and database schema once on startup."""
    global SCALER, ENCODER, RF_MODEL, SVM_MODEL, PCA_TRANSFORMER, FEATURE_MATRIX, LABEL_ARRAY, ALL_PATHS
    
    try:
        logger.info("Initializing Intelligence System...")
        SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        ENCODER = joblib.load(os.path.join(MODELS_DIR, "encoder.joblib"))
        RF_MODEL = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
        # SVM_MODEL = joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib")) # Excluded due to size
        PCA_TRANSFORMER = joblib.load(os.path.join(MODELS_DIR, "pca_transformer.joblib"))
        
        # Load all 24k features into memory from the centralized models directory
        FEATURE_MATRIX, LABEL_ARRAY, ALL_PATHS = load_feature_index(MODELS_DIR, DATASET_PATH)
        
        # Ensure directory for uploads exists
        os.makedirs(os.path.join(ROOT_DIR, "uploads"), exist_ok=True)
        
        # Initialize DB Tables
        create_tables()
        logger.info("Intelligence System Ready.")
    except Exception as e:
        logger.error(f"Startup Critical Failure: {str(e)}")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    image_id: int
    predicted_label: str
    confidence: float
    model_used: str
    timestamp: str

class SimilarityResult(BaseModel):
    rank: int
    image_id: Optional[int]
    image_path: str
    similarity_score: float
    class_label: str

class SimilarityResponse(BaseModel):
    query_source: str
    results: List[SimilarityResult]

class AnalyticsResponse(BaseModel):
    total_images: int
    class_distribution: dict
    average_confidence: float

class ImageRecord(BaseModel):
    image_id: int
    image_path: str
    class_label: str
    upload_timestamp: str

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "online", "message": "Satellite Intelligence API is active."}

@app.post("/upload", response_model=PredictionResponse)
async def upload_image(file: UploadFile = File(...), model: str = Query("rf", enum=["rf", "svm"])):
    """
    Uploads a satellite image, extracts features, predicts class, and saves result.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # 1. Temporary Save
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. Intelligence Extraction
        features = extract_features(tmp_path)
        if features is None:
            raise ValueError("Feature extraction module returned None.")

        norm_features = SCALER.transform(features.reshape(1, -1))
        # SVM is disabled in cloud due to size limits, fallback to RF
        active_model = RF_MODEL
        
        # 3. Running Inference
        norm_features = PCA_TRANSFORMER.transform(norm_features)
        pred_idx = active_model.predict(norm_features)[0]
        label = ENCODER.inverse_transform([pred_idx])[0]
        
        # Confidence calculation
        if hasattr(active_model, "predict_proba"):
            confidence = float(np.max(active_model.predict_proba(norm_features)))
        else:
            confidence = float(np.tanh(np.max(active_model.decision_function(norm_features))))

        # 4. Persistence
        uploads_dir = os.path.join(ROOT_DIR, "uploads")
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        permanent_path = os.path.join(uploads_dir, filename)
        shutil.move(tmp_path, permanent_path)

        image_id = ingest_image_pipeline(
            image_path=permanent_path,
            class_label=label,
            feature_vector=features,
            predicted_label=label,
            confidence=confidence,
            model_name=f"{model}_classifier"
        )

        return PredictionResponse(
            image_id=image_id,
            predicted_label=label,
            confidence=confidence,
            model_used=model,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Intelligence failure: {str(e)}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity", response_model=SimilarityResponse)
async def get_similarity(
    file: Optional[UploadFile] = File(None),
    image_id: Optional[int] = Query(None),
    k: int = Query(5, ge=1, le=20),
    metric: str = Query("cosine", enum=["cosine", "euclidean"])
):
    """
    Returns Top-K similar images from either an upload or a known database ID.
    """
    try:
        if file:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                query_path = tmp.name
            
            results = find_top_k_similar(
                query_image_path=query_path,
                feature_matrix=FEATURE_MATRIX,
                all_paths=ALL_PATHS,
                scaler=SCALER,
                k=k,
                metric=metric,
                pca_transformer=PCA_TRANSFORMER
            )
            os.remove(query_path)
            source = f"upload: {file.filename}"
            
        elif image_id:
            with get_connection() as conn:
                row = conn.execute("SELECT image_path FROM images WHERE image_id = ?", (image_id,)).fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Requested Image ID not found in database.")
                query_path = row["image_path"]

            results = find_top_k_similar(
                query_image_path=query_path,
                feature_matrix=FEATURE_MATRIX,
                all_paths=ALL_PATHS,
                scaler=SCALER,
                k=k,
                metric=metric,
                pca_transformer=PCA_TRANSFORMER
            )
            source = f"database_id: {image_id}"
        else:
            raise HTTPException(status_code=400, detail="Must provide either multipart 'file' or 'image_id' query param.")

        # Enrich results with DB image_ids where possible
        final_results = []
        for r in results:
            db_record = get_image_by_path(r["path"])
            final_results.append(SimilarityResult(
                rank=r["rank"],
                image_id=db_record["image_id"] if db_record else None,
                image_path=r["path"],
                similarity_score=r["score"],
                class_label=os.path.basename(os.path.dirname(r["path"]))
            ))

        return SimilarityResponse(query_source=source, results=final_results)

    except Exception as e:
        logger.error(f"Search failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images", response_model=List[ImageRecord])
def get_images(limit: int = 50, offset: int = 0):
    """Retrieves metadata history from the database."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT image_id, image_path, class_label, upload_timestamp FROM images ORDER BY upload_timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [ImageRecord(**dict(r)) for r in rows]

@app.get("/analytics", response_model=AnalyticsResponse)
def get_analytics():
    """Aggregated intelligence stats."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        dist_rows = conn.execute("SELECT class_label, COUNT(*) as count FROM images GROUP BY class_label").fetchall()
        avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0.0

    return AnalyticsResponse(
        total_images=total,
        class_distribution={r["class_label"]: r["count"] for r in dist_rows},
        average_confidence=float(avg_conf)
    )

if __name__ == "__main__":
    import uvicorn
    # Start the server (listening on all interfaces with dynamic port for Railway)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
