# 🛰️ Satellite Intelligence Dashboard 

A professional, end-to-end mission control system for satellite imagery analysis. Features real-time classification, similarity search, and geospatial mapping.

## ✨ Features
- **Real-time AI Classification**: EURO-SAT classification with 89%+ Accuracy.
- **Top-K Similarity Search**: Finds similar land-cover patterns in seconds using PCA-optimized vector search.
- **Cloud Optimized**: Multi-hundred MB models compressed and dimensionally reduced (<20MB) for seamless GitHub & Railway hosting.

---

## 🚀 Cloud Deployment Guide

### 1. Backend API (Railway.app)
1. Fork/Push this repository to GitHub.
2. In Railway, click **New Project** -> **Deploy from GitHub repo**.
3. **Custom Start Command**: 
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```
4. **Note**: The app automatically uses the `PORT` environment variable provided by Railway.

### 2. Frontend Dashboard (Streamlit Cloud)
1. Create a **New App** on Streamlit Cloud and connect your repository.
2. **Main file path**: `frontend.py`
3. **Secrets Management**: 
   In Advanced Settings, add your Railway API URL so the frontend can communicate with the backend:
   ```toml
   API_URL = "https://your-railway-app.up.railway.app"
   ```

---

## 🛠️ Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API**:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

3. **Start the Dashboard**:
   ```bash
   streamlit run frontend.py
   ```

## 📂 Project Structure
- **/app**: Core FastAPI implementation and ML inference logic.
- **/database**: SQLite persistence layer (Auto-initialized on first run).
- **/models**: PCA Transformers and compressed Joblib models for land-cover classification.
- **frontend.py**: Lightweight Streamlit UI client.
- **optimize_models.py**: Maintenance script for model compression and dimension reduction.
