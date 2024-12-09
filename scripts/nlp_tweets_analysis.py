# Import ncessary libraries
import nltk
import nltk.downloader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
from termcolor import colored

# Download necessary NLTK data
nltk.download('punkt')       # Tokenizer
nltk.download('stopwords')   # Stopword corpus
nltk.download('wordnet')     # WordNet corpus for lemmatization

# Configuration settings
FAKE_CSV_PATH = '/home/jparep/proj/nlp-tweet-analysis/data/raw/fake.csv'
REAL_CSV_PATH = '/home/jparep/proj/nlp-tweet-analysis/data/raw/true.csv'
MODEL_PATH = '/home/jparep/proj/nlp-tweet-analysis/model/model.pkl'
VECTORIZER_PATH = '/home/jparep/proj/nlp-tweet-analysis/model/vectorizer.pkl'
RANDOM_SEED = 42

# Initliaze stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

def read_csv_files(real_csv, fake_csv):
    """Load data and label fake and real and return concatenate dataframe"""
    df_fake = pd.read_csv(fake_csv)
    df_real = pd.read_csv(real_csv)
    
    df_fake['label'] = 'fake'
    df_real['label'] = 'real'
    
    df_concat = pd.concat([df_fake, df_real], axis=0).sample(frac=1).reset_index(drop=True)
    return df_concat

def preprocess_text(text):
    """Preprocess data"""
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()
    tokens = word_tokenize(text)
    lem = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lem)

def load_and_preprocess_data():
    """Load data and preprocess. Make fake and real to 1 and 0 respectively"""
    df = read_csv_files(REAL_CSV_PATH, FAKE_CSV_PATH)
    df = df[['text', 'label']]
    df['processed_text'] = df['text'].apply(preprocess_text)
    df["label"] = df["label"].map({"real": 0, "fake": 1})
    return df


# Train-validate-test split
def train_valid_test_split(X, y, train_size=0.7, valid_size=0.15, test_size=0.15):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size + test_size), random_state=RANDOM_SEED)
    
    ratio = valid_size / (valid_size + test_size)
    
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - ratio), random_state=RANDOM_SEED)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# call method
X = df['processed_text']
y = df['label']
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y)


# Vectorize data
def vectorize_data(X_train, X_valid, X_test):
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(X_train)
    xv_valid = vectorizer.fit_transform(X_valid)
    xv_test = vectorizer.fit_transform(X_test)
    
    with open(VECTORIZER_PATH, 'wb') as f:
        joblib.dump(vectorizer, f)
    
    return xv_train, xv_valid, xv_test, vectorizer

# call method
xv_train, xv_valid, xv_test, vectorizer = vectorize_data(X_train, X_valid, X_test)


# Train Model
def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

# Save Model
def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)


model = RandomForestClassifier(random_state=RANDOM_SEED)

model_trained = train_model(xv_train, y_train, model)
save_model(model_trained, MODEL_PATH)
    
    

# Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC Score: {roc_auc}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot()
    plt.show()

for name, model in models.items():
    print(f"Evaluating {name}...")
    with open(f"/home/jparep/proj/nlp-tweet-analysis/model/{name.replace(' ', '_').lower()}.pkl", 'rb') as f:
        trained_model = joblib.load(f)
    
    y_train_pred = trained_model.predict(xv_train)
    y_valid_pred = trained_model.predict(xv_valid)
    y_test_pred = trained_model.predict(xv_test)

    evaluate_model(y_train, y_train_pred, f"{name} (train)")
    evaluate_model(y_valid, y_valid_pred, f"{name} (valid)")
    evaluate_model(y_test, y_test_pred, f"{name} (test)")
    plot_confusion_matrix(y_test, y_test_pred)


# Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train):
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])

    param_distributions = {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 7, 9]
    }

    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=12, cv=10, n_jobs=-1, random_state=RANDOM_SEED, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_

optimized_model = hyperparameter_tuning(X_train, y_train)
with open(MODEL_PATH, 'wb') as f:
    joblib.dump(optimized_model, f)

print(f"Optimized model saved to {MODEL_PATH}")



# Evaluate Optimized Model
with open(MODEL_PATH, 'rb') as f:
    optimized_model = joblib.load(f)

y_train_pred = optimized_model.predict(X_train)
y_valid_pred = optimized_model.predict(X_valid)
y_test_pred = optimized_model.predict(X_test)

evaluate_model(y_train, y_train_pred, "Optimized DecisionTree (train)")
evaluate_model(y_valid, y_valid_pred, "Optimized DecisionTree (valid)")
evaluate_model(y_test, y_test_pred, "Optimized DecisionTree (test)")
plot_confusion_matrix(y_test, y_test_pred)


# Save the optimized model and vectorizer
with open(MODEL_PATH, 'wb') as f:
    joblib.dump(optimized_model, f)

with open(VECTORIZER_PATH, 'wb') as f:
    joblib.dump(vectorizer, f)
