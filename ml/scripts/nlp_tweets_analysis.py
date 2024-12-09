# Import necessary libraries
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuration settings
DIR = os.dirname(os.path.relpath(__name__))
DATA_PATH = os.path.join(DIR, 'data', 'raw')
FAKE_CSV_PATH = os.path.join(DATA_PATH, 'fake.csv')
REAL_CSV_PATH = os.path.join(DATA_PATH, 'true.csv')
MODEL_PATH = os.path.join(DIR, 'model', 'pipeline_model.joblib'
RANDOM_SEED = 42

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Utility functions for saving and loading objects
def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        joblib.dump(obj, f)

def load_object(filepath):
    with open(filepath, 'rb') as f:
        return joblib.load(f)

# Load and preprocess data
def read_csv_files(real_csv, fake_csv):
    """Load data and label fake and real news."""
    df_fake = pd.read_csv(fake_csv)
    df_real = pd.read_csv(real_csv)
    df_fake['label'] = 'fake'
    df_real['label'] = 'real'
    df = pd.concat([df_fake, df_real], axis=0).sample(frac=1).reset_index(drop=True)
    return df

def preprocess_text(text):
    """Clean and preprocess text."""
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data():
    """Load and preprocess data, returning a DataFrame."""
    df = read_csv_files(REAL_CSV_PATH, FAKE_CSV_PATH)
    df = df[['text', 'label']]
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['label'] = df['label'].map({'real': 0, 'fake': 1})
    return df

# Split data into train, validation, and test sets
def train_valid_test_split(X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
    """Split data into train, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size + test_size), random_state=RANDOM_SEED, stratify=y)
    ratio = valid_size / (valid_size + test_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - ratio), random_state=RANDOM_SEED, stratify=y_temp)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Build a pipeline
def build_pipeline():
    """Create a pipeline with TfidfVectorizer and RandomForestClassifier."""
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(random_state=RANDOM_SEED))
    ])
    return pipeline

# Evaluate model
def evaluate_model(y_true, y_pred, model_name):
    """Evaluate the model using multiple metrics."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "ROC AUC Score": roc_auc_score(y_true, y_pred)
    }
    print(pd.DataFrame(metrics, index=[0]))
    print("\n", classification_report(y_true, y_pred))
    return metrics

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Hyperparameter tuning with StratifiedKFold
def hyperparameter_tuning_with_stratified_kfold(X_train, y_train):
    """Perform hyperparameter tuning using StratifiedKFold."""
    pipeline = build_pipeline()

    param_distributions = {
        'vectorizer__max_features': [5000, 10000, None],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=10,
        cv=skf,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

# Main workflow
df = load_and_preprocess_data()
X = df['processed_text']
y = df['label']
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)

# Train and evaluate the pipeline
pipeline = build_pipeline()
pipeline.fit(X_train, y_train)
save_object(pipeline, MODEL_PATH)

y_train_pred = pipeline.predict(X_train)
y_valid_pred = pipeline.predict(X_valid)
y_test_pred = pipeline.predict(X_test)

evaluate_model(y_train, y_train_pred, "Pipeline (Train)")
evaluate_model(y_valid, y_valid_pred, "Pipeline (Valid)")
evaluate_model(y_test, y_test_pred, "Pipeline (Test)")
plot_confusion_matrix(y_test, y_test_pred)

# Hyperparameter tuning
optimized_pipeline = hyperparameter_tuning_with_stratified_kfold(X_train, y_train)
save_object(optimized_pipeline, MODEL_PATH)

# Evaluate the optimized pipeline
optimized_pipeline = load_object(MODEL_PATH)
y_test_pred = optimized_pipeline.predict(X_test)

evaluate_model(y_test, y_test_pred, "Optimized Pipeline (Test)")
plot_confusion_matrix(y_test, y_test_pred)
