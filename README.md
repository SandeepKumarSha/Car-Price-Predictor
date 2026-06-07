# 🚗 Car Price Predictor

A Machine Learning web application that predicts the selling price of a used car based on features such as company, model, year, fuel type, and kilometers driven.

The application is built using **Python, Flask, and Scikit-Learn** and is deployed online for real-time predictions.

## 🔗 Live Demo

👉 https://car-price-predictor-always.onrender.com

---
## 📸 Screenshot



![Car Price Predictor](https://github.com/SandeepKumarSha/Car-Price-Predictor/blob/056731aa8d76681b04a7e6b2075e972bc68680ab/UI%20screenshot.png)


---

## 📌 Features

- Predicts used car prices instantly
- User-friendly web interface
- Machine Learning-based prediction
- Real-time results
- Flask backend integration
- Deployed on Render

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask

### Machine Learning
- Scikit-Learn
- Pandas
- NumPy
- Joblib

### Deployment
- Gunicorn
- Render

---

## 📂 Project Structure

```text
Car-Price-Predictor
│
├── templates
│   └── index.html
│
├── app.py
├── Cleaned Car.csv
├── LinearRegressionModel.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 How It Works

1. User enters car details through the web interface.
2. Flask receives the input data.
3. The trained Linear Regression model processes the input.
4. The model predicts the estimated car price.
5. The predicted price is displayed to the user.

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/SandeepKumarSha/Car-Price-Predictor.git
cd Car-Price-Predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📦 Requirements

- Flask 3.0.3
- Gunicorn 21.2.0
- NumPy 1.26.4
- Pandas 2.0.3
- Scikit-Learn 1.6.1
- Joblib 1.3.2

---


## 👨‍💻 Author

**Sandeep Kumar Sha**

GitHub: https://github.com/SandeepKumarSha

---

⭐ If you found this project useful, consider giving it a star on GitHub.
