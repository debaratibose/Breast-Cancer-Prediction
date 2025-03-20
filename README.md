# Breast-Cancer-Prediction
This project is a Breast C💡 Key Features:
✅ Predicts breast cancer based on top medical features
✅ Built with Scikit-Learn, Flask, and Bootstrap
✅ User-friendly web app with a stylish pink theme
✅ Deployed using Flask with a beautiful background image

🛠️ Technologies Used
Python 🐍
Flask (for Web Framework) 🌐
Scikit-Learn (for Machine Learning) 🤖
Pandas & NumPy (for Data Processing) 📊
Bootstrap & CSS (for Web Design) 🎨

📂 Project Structure
Breast_Cancer_Prediction/
├── app.py                  # Flask Application
├── model.py                # Machine Learning Model Training
├── templates/
│   ├── index.html          # Webpage (User Interface)
├── static/
│   ├── styles.css          # Custom Styling
│   ├── images/
│   │   ├── cancer_banner.jpg  # Background Image
├── dataset/
│   ├── breast_cancer_data.csv  # Dataset
├── README.md               # Project Documentation

📊 Dataset Used
The dataset is based on the Wisconsin Breast Cancer Dataset (WBCD), which includes features like:
🔹 Radius Mean
🔹 Texture Mean
🔹 Perimeter Mean
🔹 Area Mean
🔹 Smoothness Mean
🔹 Compactness Mean
🔹 Concavity Mean


📈 Machine Learning Model
Algorithm Used: Logistic Regression
Feature Selection: Top 10 Features Chosen via SelectKBest
Preprocessing:
✅ Missing values handled
✅ Feature scaling with StandardScaler
✅ Train-Test Split (80-20%)

Go to: http://127.0.0.1:5000 🎀

🎯 Key Highlights
✔ Accurate Predictions using Logistic Regression
✔ Beautiful Web UI with Pink Breast Cancer Awareness Theme
✔ Easy-to-Use Web Interface for Entering Medical Data
✔ Background Image and Themed Buttons for Better User Experienceancer Prediction Web Application that uses Machine Learning (Logistic Regression) and Flask to predict whether a tumor is Malignant (Cancerous) or Benign (Non-Cancerous) based on medical features.
