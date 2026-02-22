# 1. Import Required Libraries

import pandas as pd
import numpy as np


# 2. Load the DATA SET
df = pd.read_csv("Iris.csv")
df.head()



# 3. FIND NULL VALUES
df.isnull().sum()

# 4. Finding Data Types
df.info()



# 5. DATA TRANSFORMATION

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df["Species_Encoded"] = le.fit_transform(df["Species"])
df.head()



print(df[['Species', 'Species_Encoded']].head(200))



# ------------------------------------------------------------
# STEP 6: Keep the latest dataframe (drop text column)
# ------------------------------------------------------------
df.drop(['Species','Id'], axis=1, inplace=True)   # Use inplace=True to modify the same DataFrame

print("✅ Dropped 'species' and 'Id' column successfully and using 'Species_encoded' for model training.")
df.head()


# ------------------------------------------------------------
# STEP 7: Separate features (X) and target (y)
# ------------------------------------------------------------
X = df.drop('Species_Encoded', axis=1)
y = df['Species_Encoded']

print("✅ Features and Target separated successfully!")
print("Feature Columns:", list(X.columns))
print("Target Column: Species_encoded")



# Normalization of features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Normalization done using MinMaxScaler!")
pd.DataFrame(X_scaled, columns=X.columns).head()



# ------------------------------------------------------------
# STEP 9: Split data and train Logistic Regression model
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train model
# model = LogisticRegression(max_iter=200)
model = LogisticRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully on training data!")








# ------------------------------------------------------------
# STEP 10: Evaluate model performance
# ------------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Make predictions
y_pred = model.predict(X_test)

print("📊 MODEL EVALUATION RESULTS 📊")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



y_pred = model.predict(X_test)
y_pred



# ------------------------------------------------------------
# STEP 11: Save the trained model, scaler, and encoder
# ------------------------------------------------------------
import joblib

joblib.dump(model, "iris_model.joblib")
joblib.dump(scaler, "iris_scaler.joblib")
joblib.dump(le, "iris_label_encoder.joblib")

print("💾 Model, Scaler, and Label Encoder saved successfully to current directory!")




# ------------------------------------------------------------
# STEP 12: Load saved model and predict new data
# ------------------------------------------------------------
import pandas as pd

# Load saved files
loaded_model = joblib.load("iris_model.joblib")
loaded_scaler = joblib.load("iris_scaler.joblib")
loaded_encoder = joblib.load("iris_label_encoder.joblib")

# Example new flower data: [sepal_length, sepal_width, petal_length, petal_width]
new_sample = pd.DataFrame([[4.0, 3.1, 5.0, 1.8]],
                          columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

# Normalize new data using same scaler
new_sample_scaled = loaded_scaler.transform(new_sample)

# Predict species (encoded)
pred_encoded = loaded_model.predict(new_sample_scaled)[0]

# Decode encoded value back to original species name
pred_species = loaded_encoder.inverse_transform([pred_encoded])[0]

print("🌸 NEW DATA PREDICTION 🌸")
print("Input values:", list(new_sample.values[0]))
print("Predicted Encoded Value:", pred_encoded)
print("Predicted Species Name:", pred_species)



