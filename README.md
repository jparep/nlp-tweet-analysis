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