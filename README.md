# **Sentiment Analysis on Tweets**

This project builds and deploys a sentiment analysis model for tweets using a RandomForest Classifier. The project includes text preprocessing, hyperparameter tuning, and deployment of the trained model as a Flask API on Render.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [API Deployment](#api-deployment)
- [Endpoints](#endpoints)
- [Usage Examples](#usage-examples)

---

## **Overview**

The goal of this project is to classify tweets as either positive, negative, or neutral using a machine learning pipeline. The pipeline:
1. Preprocesses tweet text (e.g., tokenization, stopword removal).
2. Trains a RandomForest Classifier.
3. Optimizes the model with hyperparameter tuning using `RandomizedSearchCV`.
4. Deploys the trained model using a Flask API.

The Flask API allows users to send a tweet as input and receive the predicted sentiment.

---

## **Features**
- Data preprocessing for tweets (tokenization, lemmatization, stopword removal).
- RandomForest Classifier for sentiment classification.
- Hyperparameter tuning with `RandomizedSearchCV` and `StratifiedKFold`.
- REST API for serving predictions using Flask.
- Deployed on Render for live API access.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: 
  - Machine Learning: `scikit-learn`
  - Text Processing: `nltk`, `pandas`
  - API: `Flask`
  - Deployment: `Render`
- **Tools**: Docker (optional for deployment)

---
## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/jparep/sentiment-analysis.git
cd sentiment-analysis
```

###  Install Dependencies

Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Load and Preprocess Data
Load the mode, preprocess and train the model.

The trained model is saved as pipeline_model.joblib in the model/ directory.
Hyperparameter Tuning

Hyperparameter tuning is done using RandomizedSearchCV. Modify the train_model.py script to adjust the parameters in the param_distributions dictionary:

param_distributions = {
    'vectorizer__max_features': [5000, 10000, None],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

Re-run the training script to train the optimized model.

### API Deployment
1. Local Testing

Run the Flask API locally:

python run.py

Access the API at:

http://127.0.0.1:5000

2. Dockerized Deployment

Build the Docker image:

docker build -t sentiment-analysis .

Run the container:

docker run -p 5000:5000 sentiment-analysis

3. Deploy on Render

    Log in to Render.
    Create a new Web Service.
    Link your GitHub repository and specify the start command:

    python scripts/flask_app.py

Endpoints
1. Health Check

URL: /health
Method: GET
Description: Check if the API is running.
Response: {"status": "API is running"}
2. Predict Sentiment

URL: /predict
Method: POST
Description: Predict the sentiment of a tweet.
Request Body:

{
    "text": "This is an example tweet."
}

Response:

{
    "sentiment": "positive"
}

#### Usage Examples
Using curl

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "I love this!"}'

Using Python

import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "I love this!"}
response = requests.post(url, json=data)
print(response.json())

#### Future Enhancements

    Support for multi-class sentiment analysis (e.g., happy, sad, angry).
    Integration with real-time tweet streams (e.g., Twitter API).
    Deployment on cloud platforms like AWS, GCP, or Azure.

Contributors

    JParep: (GitHub Profile)[https://github.com/jparep]

#### License

This project is licensed under the (MIT License)[https://opensource.org/license/mit]