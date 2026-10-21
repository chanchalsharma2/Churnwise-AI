# LIVE DEMO:https://churnwise-ai.onrender.com
# Churnwise AI - Customer Churn Prediction Dashboard
A machine learning web application built with Flask that predicts customer churn risk based on various customer attributes and provides actionable retention advice.
## 🚀 Features
- **Customer Risk Prediction:** Input customer details (contract type, monthly charges, tenure, etc.) to calculate individual churn risk percentages.
- **Risk Level Categorization:** Automatically classifies customers into **High**, **Moderate**, or **Low** risk categories.
- **Actionable Advice & Warning Factors:** Generates specific retention strategies and highlights key risk drivers for each customer.
- **Dashboard Statistics:** Displays overall dataset metrics including total customers, churn rate, and average monthly charges.
## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
- **Frontend:** HTML5, CSS3, Jinja2 Templates
## 📁 Project Directory Structure
```text
Churnwise-AI/
│
├── data/
│   └── customer_churn.csv     # Dataset used for analysis and default feature values
│
├── models/
│   └── churn_model.joblib     # Pre-trained machine learning model
│
├── static/
│   └── style.css              # Custom CSS styles for the dashboard
│
├── templates/
│   └── index.html             # Frontend user interface template
│
├── app.py                     # Main Flask application file
└── requirements.txt           # Project dependencies