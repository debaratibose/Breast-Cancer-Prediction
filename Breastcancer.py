import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
file_path = r"C:\Breast_Cancer\breast_cancer_data.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

# Convert 'diagnosis' to binary
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features & target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Select the top 10 most important features
selected_feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]  # Replace with actual top 10 features from Step 1

X_selected = X[selected_feature_names]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Normalize selected features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler & selected features
joblib.dump(scaler, "scaler.joblib")
joblib.dump(selected_feature_names, "selected_features.joblib")

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "breast_cancer_model.joblib")

print("Model trained and saved with selected features:", selected_feature_names)