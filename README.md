```markdown
# 🌱 KrishiMitra AI
**Precision Agriculture for a Sustainable Future**

KrishiMitra AI is an all-in-one agritech solution designed for farmers in West Bengal (specifically the Bardhaman region) to diagnose crop diseases, plan optimal planting strategies, and monetize carbon offsets through a "Sustainability Wallet".

GitHub Tags (Topics)
#machine-learning #agritech #streamlit #tensorflow #python #carbon-credits #west-bengal

---

## 🚀 Key Features

* **🔬 Precision Disease Diagnosis**: Uses a Custom Convolutional Neural Network (CNN) to identify 40+ crop diseases with high confidence.
* **🚜 Strategic Crop Planner**: A Random Forest Regressor that forecasts the best crop to plant based on N-P-K and Soil pH levels.
* **💰 Sustainability Wallet**: Calculates Carbon Credit accrual based on optimized fertilizer usage, allowing farmers to track environmental impact as financial value.
* **💹 Live Mandi Pulse**: Integration with Government of India APIs for real-time market rates in West Bengal.
* **💬 AI Agri-Advisor**: High-speed LLM integration (Groq Llama-3.3) for organic and chemical treatment advice.

## 📊 Dataset Repository

The complete, cleaned, and split dataset used to train the KrishiMitra AI models is hosted on Hugging Face. This includes pre-processed images for over 40 disease classes across Paddy, Tomato, Potato, and Wheat.

| Repository               | Link                                                                                    |
| :----------------------- | :---------------------------------------------------------------------------------------|
| **Hugging Face Dataset** | [🤗 iamalekhya/KrishiMitraAI](https://huggingface.co/datasets/iamalekhya/KrishiMitraAI) |

### 🛠️ Using the Data
You can load this dataset directly into your training pipeline using the `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("iamalekhya/KrishiMitraAI")


## 🧠 Model & Dataset Specifications

| Component              | Model Architecture              | Primary Dataset               | Key Metrics                 |
| :----------------------| :------------------------------ | :---------------------------- | :-------------------------- |
| **Disease Detection**  | MobileNetV2 (Transfer Learning) | PaddyDoctor & PlantVillage    | 94.2%+ Accuracy             |
| **Crop Recommendation**| Random Forest Regressor         | Soil Nutrient Dataset (India) | 99.6% Accuracy              |
| **AI Consultation**    | Llama-3.3-70B (Groq)            | Specialized Agronomy Tuning   | Ultra-Low Latency Inference |

### 📊 Dataset Details
| Dataset Name         | Description                                                   | Size / Scope |
| :--------------------| :-------------------------------------------------------------| :-------------------------------- |
| **PaddyDoctor**      | Real-world images of rice leaf diseases (Blight, Blast, etc.) | 10+ Disease Classes               |
| **PlantVillage**     | Curated images of potato, tomato, and pepper diseases         | Multi-crop global standard        |
| **Soil Health Data** | N-P-K, pH, and climate parameters for Indian soil             | Optimized for West Bengal regions |



---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Glassmorphism UI)
- **Machine Learning**: TensorFlow, Scikit-Learn
- **Language Models**: Groq Cloud (Llama-3.3-70B)
- **APIs**: OpenWeatherMap, Data.gov.in
- **Backend**: Python 3.12

---

## 📂 Project Structure

```text
📁 KrishiMitra
├── 📄 app.py                           # Main Streamlit application logic
├── 📄 requirements.txt                 # Dependency list for deployment
├── 🧠 KrishiMitra_Disease_Model.h5     # Trained CNN Model for leaf diagnosis
├── 🧠 crop_recommendation_model.pkl    # Pickled Random Forest model
├── 🖼️ background.jpg                   # UI background asset
├── 🖼️ logo.png                         # Project branding
└── 📓 Building_the_Model.ipynb         # Model training documentation

```

---

## ⚙️ Installation & Local Setup

1. **Clone the repository**:
```bash
git clone [https://github.com/AlekhyaGangopadhyay/KrishiMitra.git](https://github.com/AlekhyaGangopadhyay/KrishiMitra.git)

```


2. **Create a Virtual Environment**:
```bash
python -m venv .venv
.\.venv\Scripts\activate

```


3. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


4. **Run the App (Headless mode for stability)**:
```bash
streamlit run app.py --server.headless true

```



---

## 🔮 Future Roadmap

| Phase | Milestone | Description |
| ------------| --------------------------- | -------------------------------------------------------------------------- |
| **Phase 1** | **Multilingual Support**    | Integration of Bengali and Santali voice-to-text for rural accessibility.  |
| **Phase 2** | **IoT Integration**         | Live soil sensor (N-P-K probes) connectivity via ESP32/Arduino.            |
| **Phase 3** | **Blockchain Verification** | Deploying Carbon Credits on a decentralized ledger for transparent trading.|
| **Phase 4** | **Edge Deployment**         | Quantizing models for offline use on low-end mobile devices.               |

---

**Developed by Alekhya Gangopadhyay** *B.Tech Student, Institute of Engineering & Management (IEM)*

```


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://krishimitra-ai.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/AlekhyaGangopadhyay/KrishiMitra)

> **Live Demo:** [krishimitra-ai.streamlit.app]https://krishimitraai-otn3rk7eg2h6srn58cbf9q.streamlit.app/
---

### 💡 Why this revision is better:
1.  **Academic Depth**: The **Model & Dataset Specifications** table shows you understand the "why" behind your AI, not just the "how."
2.  **Visionary Thinking**: The **Future Roadmap** demonstrates that you see this as a scalable business or social solution, which is a major scoring criterion in hackathons.

```
---

## 🌟 Support the Project

If you found **KrishiMitra AI** helpful for your research or interesting for the future of AgTech, please consider giving it a **Star**! It helps the project reach more developers and researchers.

1. Navigate to the top of this page.
2. Click the **⭐ Star** button in the upper right corner.

---

## 🤝 Connect with Me

I am always open to discussing **Machine Learning**, **Sustainability**, or **AgTech** collaborations. Feel free to reach out!

* **📧 Email:** [iamalekhya7@gmail.com](mailto:iamalekhya7@gmail.com)
* **💼 LinkedIn:** [linkedin.com/in/iamalekhya](https://www.linkedin.com/in/alekhya-gangopadhyay-5a199a320?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
* **🤗 Hugging Face:** [huggingface.co/iamalekhya](https://huggingface.co/iamalekhya)

---
*Created with ❤️ by Alekhya Gangopadhyay in Kolkata, WB.*
