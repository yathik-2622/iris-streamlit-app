# 🌸 Iris Streamlit App

> Interactive Iris flower species predictor built with 
> Streamlit and Scikit-learn  
> By **Yathik** · **RyStudios**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)
![Streamlit Cloud](https://img.shields.io/badge/Deploy-Streamlit%20Cloud-FF4B4B?style=flat-square)

---

## 🚀 Live Demo
🌐 [rystudios-iris.streamlit.app](https://rystudios-iris.streamlit.app)

---

## ✨ Features
- 🤖 Logistic Regression model trained on Iris dataset
- 📊 Model accuracy metrics and confusion matrix
- 🌸 Real-time species prediction with probability
- 📈 Interactive feature explorer with live scatter plot
- 🗂️ Dataset sample explorer

---

## 🧭 App Sections

| Section | Description |
|---------|-------------|
| 🏠 Home | Enter measurements → get prediction |
| 📊 Model Accuracy | Performance metrics + confusion matrix |
| 🌸 Dataset Samples | Explore real Iris data samples |
| 📈 Feature Reference | Live sliders + scatter plot |
| ℹ️ About | Project summary and tech stack |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Scikit-learn | ML model |
| Pandas / NumPy | Data handling |
| Matplotlib / Seaborn | Visualization |
| Joblib | Model persistence |
| Streamlit | Frontend UI |

---

## ⚙️ Run Locally
```bash
# Clone repo
git clone https://github.com/yathik-2622/iris-streamlit-app.git
cd iris-streamlit-app

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python iris_backend.py

# Run app
streamlit run iris_frontend.py
```

---

## 📁 Folder Structure
```
iris-streamlit-app/
│
├── .streamlit/
│   └── config.toml
├── iris_backend.py            # Model training script
├── iris_frontend.py            # Streamlit frontend
├── Iris.csv                    # Dataset
├── iris_model.joblib           # Saved model
├── iris_scaler.joblib          # Saved scaler
├── iris_label_encoder.joblib   # Saved encoder
├── logo.png
├── requirements.txt
└── README.md
```

---

## 🎬 Part of RyStudios Portfolio

| App | Tech | URL |
|-----|------|-----|
| 🌸 Iris Visual AI | FastAPI + Plotly | [rystudios-iris-visual.vercel.app](https://rystudios-iris.vercel.app/) |
| 🌸 Iris Streamlit | Streamlit | [rystudios-iris.streamlit.app](https://rystudios-iris.streamlit.app)|

---

© 2025 Yathik · RyStudios. All rights reserved.