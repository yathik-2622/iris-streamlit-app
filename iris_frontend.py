import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Load saved model, scaler, encoder, dataset
# --------------------------------------------------
model = joblib.load("iris_model.joblib")
scaler = joblib.load("iris_scaler.joblib")
encoder = joblib.load("iris_label_encoder.joblib")
df = pd.read_csv("Iris.csv")

# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------
# st.set_page_config(page_title="🌸 Iris Flower App", page_icon="🌺", layout="wide")
st.set_page_config(
    page_title="Iris | RyStudios",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("🔍 Navigation Menu")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Model Accuracy", "🌸 Dataset Samples", "📈 Feature Reference", "ℹ️ About App"],
    index=0
)

# --------------------------------------------------
# Helper Function: Compact Confusion Matrix
# --------------------------------------------------
def show_confusion_matrix():
    X = df.drop(columns=["Id", "Species"])
    y = encoder.transform(df["Species"])  # Use saved encoder
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("✅ Model Accuracy Summary")
    st.metric("Overall Accuracy", f"{acc*100:.2f}%")

    col1, col2 = st.columns([0.6, 0.4], gap="small")
    with col1:
        st.write("### Confusion Matrix Visualization")
        fig, ax = plt.subplots(figsize=(3, 2.5))  # reduced size
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=encoder.inverse_transform([0, 1, 2]),
            yticklabels=encoder.inverse_transform([0, 1, 2])
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("""
        ### 🧩 Understanding the Confusion Matrix
        - ✅ **Diagonal cells**: Correct predictions  
        - ❌ **Off-diagonal cells**: Misclassifications  
        - Most overlap occurs between *Versicolor* and *Virginica*  
        - A perfect model → all numbers on diagonal  
        """)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    st.title("🌼 Iris Flower Species Prediction")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("📥 Enter Flower Measurements")
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width  = st.number_input("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width  = st.number_input("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=0.2, step=0.1)

        predict_btn = st.button("🔍 Predict Species")

    with col2:
        st.subheader("📊 Prediction Result")

        if predict_btn:
            sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
            sample_scaled = scaler.transform(sample)
            pred_encoded = model.predict(sample_scaled)[0]
            pred_species = encoder.inverse_transform([pred_encoded])[0]

            probs = model.predict_proba(sample_scaled)[0]
            prob_df = pd.DataFrame({
                "Species": encoder.inverse_transform([0, 1, 2]),
                "Confidence (%)": np.round(probs * 100, 2)
            }).sort_values(by="Confidence (%)", ascending=False)

            st.success(f"🌸 **Predicted Species:** {pred_species}")
            st.markdown("#### 🔢 Model Confidence")
            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

            st.markdown("#### 🧾 Input Summary")
            st.dataframe(sample, use_container_width=True)
        else:
            st.info("👈 Enter values and click *Predict Species* to get results.")

# --------------------------------------------------
# MODEL ACCURACY PAGE
# --------------------------------------------------
elif page == "📊 Model Accuracy":
    st.title("📊 Model Accuracy & Confusion Matrix")
    show_confusion_matrix()

# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------
elif page == "🌸 Dataset Samples":
    st.title("🌺 Iris Dataset Samples")
    st.markdown("Here are 20 randomly selected samples from the Iris dataset:")
    st.dataframe(df.sample(20, random_state=42), use_container_width=True)

# --------------------------------------------------
# FEATURE REFERENCE PAGE (compact layout)
# --------------------------------------------------
elif page == "📈 Feature Reference":
    st.title("📈 Feature Ranges & Live Comparison")

    st.markdown("Use the sliders below to explore how changes affect predictions:")

    # --- Two-column layout for balance ---
    left_col, right_col = st.columns([0.45, 0.55], gap="small")

    with left_col:
        s_len = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
        s_wid = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        p_len = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        p_wid = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

        sample = pd.DataFrame([[s_len, s_wid, p_len, p_wid]],
                              columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
        scaled = scaler.transform(sample)
        pred_encoded = model.predict(scaled)[0]
        pred_species = encoder.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba(scaled)[0]

        st.markdown(f"### 🌸 Predicted Species: **{pred_species}**")
        st.progress(float(max(probs)))
        st.dataframe(pd.DataFrame({
            "Species": encoder.inverse_transform([0, 1, 2]),
            "Confidence (%)": np.round(probs * 100, 2)
        }).sort_values(by="Confidence (%)", ascending=False),
        use_container_width=True)

    with right_col:
        st.markdown("#### 📊 Visual Comparison (Petal Length vs Width)")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.scatterplot(
            data=df,
            x="PetalLengthCm",
            y="PetalWidthCm",
            hue="Species",
            palette="husl",
            alpha=0.7,
            s=50
        )
        ax.scatter(p_len, p_wid, color="black", s=100, label="Your Input", edgecolor="white")
        ax.legend(fontsize=8)
        plt.title("Your Input vs Dataset", fontsize=10)
        st.pyplot(fig, use_container_width=True)

    st.info("""
    💡 *Observation:*  
    The black dot shows your current input.  
    Adjust sliders and see how it moves between species clusters.
    """)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
# elif page == "ℹ️ About App":
#     st.title("ℹ️ About This Project")
#     st.markdown("""
#     **Model:** Logistic Regression  
#     **Framework:** Streamlit  
#     **Dataset:** UCI Iris Dataset  

#     This project demonstrates interactive ML predictions with:
#     - 📊 Confidence visualization  
#     - 🎛 Dynamic feature controls  
#     - 🧠 Real-time species explanation  
#     """)
#     st.success("Use the sidebar to explore different sections — no restarts needed 🚀")

# ABOUT PAGE
# --------------------------------------------------
elif page == "ℹ️ About App":
    st.title("ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model:** Logistic Regression  
        **Framework:** Streamlit  
        **Dataset:** UCI Iris Dataset  
        **Deployed:** Streamlit Cloud  
        """)
    
    with col2:
        st.markdown("""
        **Built by:** Yathik  
        **Brand:** RyStudios  
        **GitHub:** [iris-streamlit-app](https://github.com/yathik-2622/iris-streamlit-app)  
        **Live:** [rystudios-iris.streamlit.app](https://rystudios-iris.streamlit.app)  
        """)
    
    st.markdown("---")
    st.markdown("""
    This app demonstrates end-to-end ML deployment with:
    - 📊 Real-time confidence visualization
    - 🎛️ Dynamic feature controls  
    - 🧠 Live species prediction with explanation
    - 📈 Interactive scatter plots
    - 🔬 Model accuracy and confusion matrix
    """)
    st.markdown("---")
    st.markdown("© 2025 **Yathik** · **RyStudios** — Where Data Meets Design 🎬")
    
    st.success("Use the sidebar to explore different sections🚀")