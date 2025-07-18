# 📰 Fake News Detection App

This is a **Streamlit web app** that uses machine learning (Logistic Regression) to classify news articles as **FAKE** or **REAL** based on their text content.

## 🚀 Features

- Upload a ZIP file containing `Fake.csv` and `True.csv`
- Trains a Logistic Regression model on the uploaded data
- Displays model accuracy and confusion matrix
- Allows users to classify custom news articles
- Uses TF-IDF, NLTK for text cleaning, and Scikit-learn for ML

## 📦 ZIP File Format

The uploaded ZIP file **must contain**:
- `Fake.csv`
- `True.csv`

Each file should have at least the column `text`.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## 📋 Installation

```bash
# Clone the repo
git clone https://github.com/Ramen4611/Fake-News-Detection-System.git
cd Fake-News-Detection-System

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
