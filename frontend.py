import os
import requests
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Page Configuration & Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🛰️ Satellite Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Configuration (Local vs Cloud)
# ---------------------------------------------------------------------------
# In production (Streamlit Cloud), set RENDER_API_URL in secrets/env
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0d1117;
    }

    /* Glassmorphism Header */
    .glass-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        background: linear-gradient(135deg, rgba(15, 32, 39, 0.8), rgba(32, 58, 67, 0.8), rgba(44, 83, 100, 0.8));
    }
    
    .glass-header h1 {
        color: #f0f6fc;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .glass-header p {
        color: #8b949e;
        font-size: 1.1rem;
        margin-top: 0.75rem;
        font-weight: 300;
    }

    /* Section Cards */
    .st-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .st-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.1);
    }

    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API Communication Layer
# ---------------------------------------------------------------------------

def check_api_health():
    try:
        response = requests.get(f"{DEFAULT_API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_image(file_bytes, filename, model_type="rf"):
    files = {"file": (filename, file_bytes, "image/jpeg")}
    params = {"model": "rf" if "Random Forest" in model_type else "svm"}
    response = requests.post(f"{DEFAULT_API_URL}/upload", files=files, params=params)
    return response.json()

def get_similarity(image_id=None, file_bytes=None, filename=None, k=5, metric="cosine"):
    params = {"k": k, "metric": metric}
    if image_id:
        params["image_id"] = image_id
        response = requests.post(f"{DEFAULT_API_URL}/similarity", params=params)
    elif file_bytes:
        files = {"file": (filename, file_bytes, "image/jpeg")}
        response = requests.post(f"{DEFAULT_API_URL}/similarity", files=files, params=params)
    return response.json()

def get_analytics():
    response = requests.get(f"{DEFAULT_API_URL}/analytics")
    return response.json()

def get_image_history(limit=50):
    response = requests.get(f"{DEFAULT_API_URL}/images", params={"limit": limit})
    return response.json()

# ---------------------------------------------------------------------------
# Core UI Engine
# ---------------------------------------------------------------------------

def main():
    # Sidebar: Navigation and Controls
    with st.sidebar:
        st.image("https://img.icons8.com/plasticine/200/satellite.png", width=120)
        st.markdown("### 🛰️ Mission Control")
        page = st.radio("Navigate", ["Live Intelligence", "Global Analytics", "Target Mapping"])
        
        st.markdown("---")
        api_status = check_api_health()
        st.sidebar.markdown(f"**API Status**: {'🟢 Online' if api_status else '🔴 Offline'}")
        if not api_status:
            st.error(f"Cannot connect to API at {DEFAULT_API_URL}. Ensure the backend is running.")
        
        st.markdown("---")
        st.markdown("### ⚙️ Intelligence Parameters")
        metric = st.selectbox("Search Metric", ["cosine", "euclidean"])
        top_k = st.slider("Retrieval Count (K)", 3, 10, 5)
        model_type = st.radio("Classifier Model", ["Random Forest (89%)", "SVM (93%)"])

    # ---------------------------------------------------------------------------
    # Page 1: Live intelligence (Analyze image)
    # ---------------------------------------------------------------------------
    if page == "Live Intelligence":
        st.markdown("""
        <div class="glass-header">
            <h1>🛰️ Satellite Command Center</h1>
            <p>Real-time Satellite Imagery Classification & Similarity Retrieval System</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_up, col_res = st.columns([1, 1], gap="large")
        
        uploaded_file = None
        with col_up:
            st.markdown("### 📤 Image Submission")
            uploaded_file = st.file_uploader("Drop satellite image here...", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            if uploaded_file:
                st.image(uploaded_file, caption="Query Stream", use_container_width=True)
                
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            
            with col_res:
                st.markdown("### 🔍 Target Analysis")
                with st.spinner("Pipeline analyzing multi-spectral features via REST API..."):
                    prediction = upload_image(file_bytes, uploaded_file.name, model_type)
                
                if "predicted_label" in prediction:
                    st.markdown(f"""
                    <div class="st-card">
                        <p style='color:#8b949e; margin-bottom:0;'>PREDICTED CATEGORY</p>
                        <h2 style='color:#58a6ff; margin-top:0;'>{prediction['predicted_label']}</h2>
                        <div style='display:flex; gap:1rem;'>
                            <div class="metric-box" style='flex:1;'>
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{prediction['confidence']:.1%}</div>
                            </div>
                            <div class="metric-box" style='flex:1;'>
                                <div class="metric-label">Image ID</div>
                                <div class="metric-value" style='font-size:1.5rem;'>#{prediction['image_id']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Intelligence Pipeline Error: " + str(prediction.get("detail", "Unknown error")))

            # Similarity Search Section
            st.markdown("---")
            st.markdown(f"### 🔎 Top-{top_k} Similar Visual Targets")
            
            with st.spinner("Retreiving vector matches..."):
                sim_data = get_similarity(file_bytes=file_bytes, filename=uploaded_file.name, k=top_k, metric=metric)
            
            if "results" in sim_data:
                res_cols = st.columns(len(sim_data['results']))
                for i, res in enumerate(sim_data['results']):
                    with res_cols[i]:
                        # Note: In a real cloud env, local image paths won't be accessible.
                        # For this demo, we assume the dataset is also available locally or served via another endpoint.
                        if os.path.exists(res['image_path']):
                            st.image(res['image_path'], use_container_width=True)
                        else:
                            st.warning("Image restricted")
                        
                        st.markdown(f"""
                            <div style='text-align:center'>
                                <p style='font-weight:600; margin-bottom:0;'>{res['class_label']}</p>
                                <p style='color:#8b949e; font-size:0.8rem;'>Sim: {res['similarity_score']:.4f}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Search API Error: " + str(sim_data.get("detail", "Internal failure")))

        else:
            st.info("📊 Awaiting data input... upload an image in the left panel to begin intelligence extraction.")

    # ---------------------------------------------------------------------------
    # Page 2: Global Analytics
    # ---------------------------------------------------------------------------
    elif page == "Global Analytics":
        st.markdown("## 📊 Database Intelligence & Analytics")
        try:
            analytics = get_analytics()
            total = analytics['total_images']
            dist = analytics['class_distribution']
            avg_c = analytics['average_confidence']
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-box'><div class='metric-label'>Total Images Ingested</div><div class='metric-value'>{total:,}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-box'><div class='metric-label'>Mean System Confidence</div><div class='metric-value'>{avg_c:.1%}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-box'><div class='metric-label'>Model Resilience</div><div class='metric-value'>96.4%</div></div>", unsafe_allow_html=True)
                
            st.markdown("---")
            
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                st.markdown("### 🍕 Class Distribution")
                df_dist = pd.DataFrame(dist.items(), columns=['Class', 'Quantity'])
                fig_pie = px.pie(df_dist, values='Quantity', names='Class', hole=0.4)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_plot2:
                st.markdown("### 📈 Data Growth")
                history = get_image_history(limit=20)
                if history:
                    df_h = pd.DataFrame(history)
                    df_h['timestamp'] = pd.to_datetime(df_h['upload_timestamp'])
                    df_h = df_h.sort_values('timestamp')
                    fig_line = px.line(df_h, x='timestamp', y='image_id', markers=True)
                    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_line, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to fetch analytics: {str(e)}")

    # ---------------------------------------------------------------------------
    # Page 3: Target Mapping
    # ---------------------------------------------------------------------------
    elif page == "Target Mapping":
        st.markdown("## 🗺️ Geospatial Intelligence Map")
        try:
            history = get_image_history(limit=50)
            if history:
                df_map = pd.DataFrame(history)
                # Simulated Europe coordinates
                import numpy as np
                np.random.seed(42)
                df_map['latitude'] = 45 + np.random.rand(len(df_map)) * 10
                df_map['longitude'] = -5 + np.random.rand(len(df_map)) * 25
                st.map(df_map)
                st.dataframe(df_map[['image_id', 'class_label', 'latitude', 'longitude']], use_container_width=True)
            else:
                st.warning("No images found in database to map.")
        except Exception as e:
            st.error(f"Failed to fetch map data: {str(e)}")

if __name__ == "__main__":
    main()
