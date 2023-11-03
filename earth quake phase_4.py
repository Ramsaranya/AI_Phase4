# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load your dataset (fake and real news articles) into a Pandas DataFrame
data = pd.read_csv('your_dataset.csv')

# Text Preprocessing
# Tokenization, Lowercasing, Stopword Removal, and Lemmatization
# You might need to use NLTK or spaCy for this step
# Below is a simple example with NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['text'] = data['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = data['text']
y = data['label']  # 1 for fake, 0 for real

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model Training
# In this example, we'll use a simple Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy and other evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

# Hyperparameter Tuning using GridSearchCV
# Example of hyperparameter tuning for a different classifier, e.g., SVM
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_model = grid_search.best_estimator_

# Deploy your model for real-world usage, e.g., using a REST API or a web application
