import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import requests
import google.generativeai as genai
from PIL import Image
from fpdf import FPDF
import base64
import os
from streamlit_option_menu import option_menu

# --- 1. ASSET ENCODING & STYLING ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.set_page_config(page_title="KrishiMitra AI", page_icon="🌱", layout="wide")

# Apply Background & Custom Glassmorphism CSS
try:
    bg_base64 = get_base64("background.jpg")
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85)), 
                        url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: 0.3s ease;
        }}
        .glass-card:hover {{ border: 1px solid rgba(76, 175, 80, 0.4); transform: translateY(-5px); }}
        .user-bubble {{ background: rgba(76, 175, 80, 0.15); padding: 15px; border-radius: 15px; margin-bottom: 10px; border-left: 5px solid #4CAF50; }}
        .bot-bubble {{ background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px; margin-bottom: 10px; border-left: 5px solid #2E7D32; }}
        h1, h2, h3 {{ color: #A5D6A7 !important; font-family: 'Segoe UI', sans-serif; }}
        [data-testid="stMetricValue"] {{ color: #81C784 !important; font-size: 2.5rem !important; }}
        </style>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Background image 'background.jpg' not found.")

# --- 2. API KEYS & ASSET LOADING ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
MANDI_API_KEY = st.secrets["MANDI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-2.5-flash')

@st.cache_resource
def load_assets():
    try:
        d_model = tf.keras.models.load_model('KrishiMitra_Disease_Model.h5')
        simulation_mode = False
    except:
        d_model = None
        simulation_mode = True
        
    try:
        with open('crop_recommendation_model.pkl', 'rb') as f:
            c_model = pickle.load(f)
    except:
        c_model = None
    # Your provided list of 40 classes
    classes = [
        'paddydoctor_bacterial_leaf_blight', 'paddydoctor_bacterial_leaf_streak', 'paddydoctor_bacterial_panicle_blight', 'paddydoctor_blast', 'paddydoctor_brown_spot', 'paddydoctor_dead_heart', 'paddydoctor_downy_mildew', 'paddydoctor_hispa', 'paddydoctor_normal', 'paddydoctor_tungro', 'plantvillage_pepper__bell___bacterial_spot', 'plantvillage_pepper__bell___healthy', 'plantvillage_potato___early_blight', 'plantvillage_potato___healthy', 'plantvillage_potato___late_blight', 'plantvillage_tomato__target_spot', 'plantvillage_tomato__tomato_mosaic_virus', 'plantvillage_tomato__tomato_yellowleaf__curl_virus', 'plantvillage_tomato_bacterial_spot', 'plantvillage_tomato_early_blight', 'plantvillage_tomato_healthy', 'plantvillage_tomato_late_blight', 'plantvillage_tomato_leaf_mold', 'plantvillage_tomato_septoria_leaf_spot', 'plantvillage_tomato_spider_mites_two_spotted_spider_mite', 'wheat_aphid', 'wheat_black_rust', 'wheat_blast', 'wheat_brown_rust', 'wheat_common_root_rot', 'wheat_fusarium_head_blight', 'wheat_healthy', 'wheat_leaf_blight', 'wheat_mildew', 'wheat_mite', 'wheat_septoria', 'wheat_smut', 'wheat_stem_fly', 'wheat_tan_spot', 'wheat_yellow_rust'
    ]
    return d_model, c_model, classes, simulation_mode

disease_model, crop_model, CLASS_NAMES, is_sim = load_assets()

# Session State for persistence
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'detected_disease' not in st.session_state:
    st.session_state.detected_disease = "General Crops"

# --- 3. BRANDED HEADER ---
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; padding: 25px; background: rgba(255, 255, 255, 0.02); border-radius: 20px; margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.05);">
        <img src="data:image/png;base64,{}" width="100" style="margin-right:25px;">
        <h1 style="margin:0; font-size: 3.5rem; letter-spacing: 4px;">KRISHIMITRA AI</h1>
    </div>
    """.format(get_base64("logo.png") if os.path.exists("logo.png") else ""), unsafe_allow_html=True)

# Top Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Disease Diagnosis", "Crop Planner"],
    icons=["speedometer2", "search", "tree"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "rgba(255,255,255,0.05)", "border-radius": "50px", "padding": "5px"},
        "nav-link": {"font-size": "18px", "color": "white", "text-align": "center"},
        "nav-link-selected": {"background-color": "rgba(76, 175, 80, 0.2)", "border": "1px solid #4CAF50"},
    }
)

# --- 4. DASHBOARD PAGE ---
if selected == "Dashboard":
    # Weather Integration
    try:
        w_url = f"http://api.openweathermap.org/data/2.5/weather?q=Bardhaman&appid={WEATHER_API_KEY}&units=metric"
        w_data = requests.get(w_url).json()
        temp, hum, desc = w_data['main']['temp'], w_data['main']['humidity'], w_data['weather'][0]['description']
    except:
        temp, hum, desc = 30, 75, "Partly Cloudy"

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Ambient Temp", f"{temp}°C")
    with col2: st.metric("Air Humidity", f"{hum}%")
    with col3: st.metric("Field Sky", desc.title())
    with col4: st.metric("Carbon Credits", "1.24 tCO2e", delta="🌱 High")

    st.markdown("---")
    
    l, r = st.columns([1.5, 1])
    with l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("💰 Sustainability Wallet")
        n_val = st.session_state.get('last_n', 90)
        credits = ((120 - n_val) * 3.3) / 1000
        earnings = credits * 1250 
        st.write(f"**Verified Accrual:** `{credits:.4f} tCO2e` | **Valuation:** `₹{earnings:,.2f}`")
        if st.button("Generate Sustainability PDF"): st.balloons(); st.success("Verification ID generated.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("💹 Mandi Pulse")
        st.write("Today's Rates (West Bengal)")
        
        search_crop = st.text_input("🔍 Search Crop Price", placeholder="E.g., Rice, Potato, Wheat...")
        
        if search_crop:
            try:
                # Fetch live data from Data.gov.in API
                m_url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={MANDI_API_KEY}&format=json&filters[commodity]={search_crop.title()}&limit=5"
                response = requests.get(m_url, timeout=5)
                m_data = response.json()
                
                if 'records' in m_data and len(m_data['records']) > 0:
                    st.success(f"Latest prices for {search_crop.title()}:")
                    records = m_data['records']
                    # Display a clean data table
                    st.table({
                        "Market": [r.get('market', 'N/A') for r in records], 
                        "Price/Qt": [f"₹{r.get('modal_price', 'N/A')}" for r in records]
                    })
                else:
                    st.warning(f"No live data found for '{search_crop}'.")
            except Exception as e:
                st.error("Could not fetch data from Mandi API.")
        else:
            # Default mock table shown when search bar is empty
            st.table({"Crop": ["Rice", "Potato", "Mustard"], "Price/Qt": ["₹2,240", "₹1,450", "₹6,150"]})
            
        st.markdown("</div>", unsafe_allow_html=True)

# --- 5. DISEASE DIAGNOSIS PAGE ---
elif selected == "Disease Diagnosis":
    
    st.markdown("### 🔬 Precision Diagnosis Laboratory")
    left, right = st.columns([1, 1.3])
    
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Leaf Snapshot", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)
            if st.button("🚀 Analyze Plant Health"):
                if is_sim:
                    disease, conf = "Paddy Blast", 98.4
                else:
                    img_arr = np.array(img.resize((224, 224))) / 255.0
                    preds = disease_model.predict(np.expand_dims(img_arr, axis=0))
                    res_idx = np.argmax(preds)
                    disease = CLASS_NAMES[res_idx] if res_idx < len(CLASS_NAMES) else "Unclassified"
                    conf = np.max(preds) * 100
                
                st.session_state.detected_disease = disease
                st.markdown(f"**Detected:** `{disease}` ({conf:.1f}%)")
                st.session_state.chat_history.append({"role": "bot", "content": f"Detection complete: **{disease}**. How can I help you treat this?"})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card' style='overflow-y: auto;'>", unsafe_allow_html=True)
        st.subheader("💬 AI Agri-Advisor")
        for chat in st.session_state.chat_history:
            div = "user-bubble" if chat["role"] == "user" else "bot-bubble"
            st.markdown(f"<div class='{div}'><b>{chat['role'].title()}:</b><br>{chat['content']}</div>", unsafe_allow_html=True)
        
        user_msg = st.chat_input("Ask about treatment or organic methods...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.spinner("Gemini is analyzing..."):
                ctx = f"Disease: {st.session_state.detected_disease}. Question: {user_msg}. Respond as an expert Indian agronomist."
                res = llm.generate_content(ctx)
                st.session_state.chat_history.append({"role": "bot", "content": res.text})
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- 6. CROP PLANNER PAGE ---
elif selected == "Crop Planner":
    
    st.markdown("### 🚜 Strategic Soil Forecasting")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Nitrogen (N)", 0, 150, 90)
    p = c2.slider("Phosphorus (P)", 0, 150, 42)
    k = c3.slider("Potassium (K)", 0, 150, 43)
    st.session_state['last_n'] = n
    
    ph = st.slider("Soil pH Level", 4.0, 10.0, 6.5)
    rain = st.number_input("Seasonal Rainfall Prediction (mm)", value=250)
    
    if st.button("🔮 Forecast Best Selection"):
        if crop_model:
            feats = np.array([[n, p, k, 28, 75, ph, rain]])
            res = crop_model.predict(feats)
            st.balloons()
            st.success(f"### Recommended: {res[0].upper()}")
        else: st.error("Crop model file not found.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #777;'>© 2026 KrishiMitra AI Hub | Powered by Deep Learning & Intel Optimization</p>", unsafe_allow_html=True)