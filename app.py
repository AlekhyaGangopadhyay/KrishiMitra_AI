import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import requests
import base64
import os
from groq import Groq  # ✅ Groq SDK
from PIL import Image
from fpdf import FPDF
from streamlit_option_menu import option_menu

# --- 1. ASSET ENCODING & STYLING ---
def get_base64(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.set_page_config(page_title="KrishiMitra AI", page_icon="🌱", layout="wide")

# Custom Glassmorphism CSS
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
    }}
    .user-bubble {{ background: rgba(76, 175, 80, 0.15); padding: 15px; border-radius: 15px; margin-bottom: 10px; border-left: 5px solid #4CAF50; }}
    .bot-bubble {{ background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px; margin-bottom: 10px; border-left: 5px solid #2E7D32; }}
    h1, h2, h3 {{ color: #A5D6A7 !important; }}
    [data-testid="stMetricValue"] {{ color: #81C784 !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. API KEYS & CLIENT SETUP ---
# Fetching keys from Streamlit Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
    MANDI_API_KEY = st.secrets["MANDI_API_KEY"]
except KeyError as e:
    st.error(f"Missing Secret Key: {e}. Please add it to .streamlit/secrets.toml")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"  # High-speed, high-intelligence model

@st.cache_resource
def load_assets():
    # Load Disease Model
    try:
        d_model = tf.keras.models.load_model('KrishiMitra_Disease_Model.h5')
        simulation_mode = False
    except:
        d_model = None
        simulation_mode = True
        
    # Load Crop Model
    try:
        with open('crop_recommendation_model.pkl','rb') as f:
            c_model = pickle.load(f)
    except:
        c_model = None

    classes = ['paddydoctor_bacterial_leaf_blight', 'paddydoctor_bacterial_leaf_streak', 'paddydoctor_bacterial_panicle_blight', 'paddydoctor_blast', 'paddydoctor_brown_spot', 'paddydoctor_dead_heart', 'paddydoctor_downy_mildew', 'paddydoctor_hispa', 'paddydoctor_normal', 'paddydoctor_tungro', 'plantvillage_pepper__bell___bacterial_spot', 'plantvillage_pepper__bell___healthy', 'plantvillage_potato___early_blight', 'plantvillage_potato___healthy', 'plantvillage_potato___late_blight', 'plantvillage_tomato__target_spot', 'plantvillage_tomato__tomato_mosaic_virus', 'plantvillage_tomato__tomato_yellowleaf__curl_virus', 'plantvillage_tomato_bacterial_spot', 'plantvillage_tomato_early_blight', 'plantvillage_tomato_healthy', 'plantvillage_tomato_late_blight', 'plantvillage_tomato_leaf_mold', 'plantvillage_tomato_septoria_leaf_spot', 'plantvillage_tomato_spider_mites_two_spotted_spider_mite', 'wheat_aphid', 'wheat_black_rust', 'wheat_blast', 'wheat_brown_rust', 'wheat_common_root_rot', 'wheat_fusarium_head_blight', 'wheat_healthy', 'wheat_leaf_blight', 'wheat_mildew', 'wheat_mite', 'wheat_septoria', 'wheat_smut', 'wheat_stem_fly', 'wheat_tan_spot', 'wheat_yellow_rust']


    return d_model, c_model, classes, simulation_mode

disease_model, crop_model, CLASS_NAMES, is_sim = load_assets()

# State Management
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'detected_disease' not in st.session_state:
    st.session_state.detected_disease = "General Crops"

# --- 3. BRANDED HEADER ---
logo_b64 = get_base64("logo.png")
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 20px; background: rgba(255, 255, 255, 0.02); border-radius: 20px; margin-bottom: 30px;">
        <img src="data:image/png;base64,{logo_b64}" width="80" style="margin-right:20px;">
        <h1 style="margin:0; font-size: 3rem;">KrishiMitra AI</h1>
    </div>
    """, unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Disease Diagnosis", "Crop Planner"],
    icons=["speedometer2", "search", "tree"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "rgba(255,255,255,0.05)", "border-radius": "50px"},
        "nav-link-selected": {"background-color": "#4CAF50"},
    }
)

# --- 4. DASHBOARD PAGE ---
if selected == "Dashboard":
    try:
        w_url = f"http://api.openweathermap.org/data/2.5/weather?q=Bardhaman&appid={WEATHER_API_KEY}&units=metric"
        w_data = requests.get(w_url).json()
        temp, hum, desc = w_data['main']['temp'], w_data['main']['humidity'], w_data['weather'][0]['description']
    except:
        temp, hum, desc = 28, 65, "Clear Sky"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temp", f"{temp}°C")
    col2.metric("Humidity", f"{hum}%")
    col3.metric("Sky", desc.title())
    col4.metric("Carbon Credits", "1.24", delta="🌱")

    st.markdown("---")
    l, r = st.columns([1.5, 1])
    
    with l:
        st.markdown("<div class='glass-card'><h3>💰 Sustainability Wallet</h3>", unsafe_allow_html=True)
        n_val = st.session_state.get('last_n', 90)
        credits = ((150 - n_val) * 2.5) / 1000
        st.write(f"Estimated Carbon Accrual: **{credits:.4f} tCO2e**")
        st.button("Withdraw Credits")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with r:
        st.markdown("<div class='glass-card'><h3>💹 Mandi Pulse</h3>", unsafe_allow_html=True)
        search_crop = st.text_input("Search Price", placeholder="E.g. Potato")
        if search_crop:
            st.info(f"Fetching latest rates for {search_crop}...")
            # Placeholder for actual API logic
            st.table({"Market": ["Bardhaman", "Kolkata"], "Price": ["₹1,400", "₹1,550"]})
        st.markdown("</div>", unsafe_allow_html=True)

# --- 5. DISEASE DIAGNOSIS PAGE ---
elif selected == "Disease Diagnosis":
    st.markdown("### 🔬 Precision Diagnosis")
    left, right = st.columns([1, 1.3])
    
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Leaf Photo", type=['jpg', 'png'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)
            if st.button("Analyze Health"):
                if is_sim:
                    disease, conf = "Wheat Yellow Rust", 94.2
                else:
                    # ML Logic
                    img_arr = np.array(img.resize((224, 224))) / 255.0
                    preds = disease_model.predict(np.expand_dims(img_arr, axis=0))
                    disease = CLASS_NAMES[np.argmax(preds)]
                    conf = np.max(preds) * 100
                
                st.session_state.detected_disease = disease
                st.success(f"Result: {disease} ({conf:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'><h3>💬 AI Agri-Advisor (Groq)</h3>", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            role_class = "user-bubble" if chat["role"] == "user" else "bot-bubble"
            st.markdown(f"<div class='{role_class}'><b>{chat['role'].upper()}:</b><br>{chat['content']}</div>", unsafe_allow_html=True)
        
        user_input = st.chat_input("Ask about fertilizers, organic cures...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # --- GROQ API CALL ---
            prompt = f"Expert Indian Agronomist Mode. Crop Disease: {st.session_state.detected_disease}. Question: {user_input}. Provide organic and chemical solutions available in India."
            
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are KrishiMitra, a helpful AI expert in Indian Agriculture."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            ans = completion.choices[0].message.content
            st.session_state.chat_history.append({"role": "bot", "content": ans})
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- 6. CROP PLANNER PAGE ---
elif selected == "Crop Planner":
    st.markdown("### 🚜 Soil Forecasting")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Nitrogen", 0, 150, 80)
    p = c2.slider("Phosphorus", 0, 150, 40)
    k = c3.slider("Potassium", 0, 150, 40)
    ph = st.slider("pH Level", 4.0, 9.0, 6.5)
    
    if st.button("Forecast Best Crop"):
        if crop_model:
            # Note: Features depend on your specific model training (N, P, K, Temp, Hum, pH, Rain)
            features = np.array([[n, p, k, 25, 70, ph, 200]]) 
            prediction = crop_model.predict(features)
            st.balloons()
            st.success(f"Recommended Crop: {prediction[0].upper()}")
        else:
            st.warning("Model file 'crop_recommendation_model.pkl' not found.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #888;'>© 2026 KrishiMitra AI | Powered by Groq Llama-3.3</p>", unsafe_allow_html=True)