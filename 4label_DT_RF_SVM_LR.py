# Import necessary libraries
import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Set random seeds for reproducibility
np.random.seed(42)

# Load and shuffle the dataset
df = pd.read_csv('SC_4label.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Download necessary NLTK data
nltk.download('punkt')

# Tokenize the code
df['code_tokens'] = df['code'].apply(word_tokenize)

# Train Word2Vec model with a fixed seed
word2vec_model_code = Word2Vec(df['code_tokens'], vector_size=200, window=15, min_count=5, workers=10, seed=42)

# Define function to get word embeddings
def get_word_embeddings(tokens, word2vec_model):
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:  # Check if token is in the vocabulary
            embeddings.append(word2vec_model.wv[token])
    return embeddings

# Apply word embeddings to the code tokens
df['code_embeddings'] = df['code_tokens'].apply(lambda x: get_word_embeddings(x, word2vec_model_code))

# Compute average embeddings for each code snippet
df['code_avg_embedding'] = df['code_embeddings'].apply(lambda x: sum(x) / len(x))

# Convert embeddings to DataFrame
code_embeddings = pd.DataFrame(df['code_avg_embedding'].to_list())

# Scale the embeddings
scaler = MinMaxScaler()
code_embeddings_scaled = scaler.fit_transform(code_embeddings)

# Assuming you have a target variable 'label_encoded' in your DataFrame
X = pd.concat([pd.DataFrame(code_embeddings_scaled)], axis=1)
y = df['label_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers with fixed random states
classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=40),
    "Random Forest": RandomForestClassifier(random_state=40, max_depth=20, min_samples_split=2, n_estimators=200),
    "SVM": SVC(random_state=40, C=100, gamma=1, kernel='rbf'),
    "Logistic Regression": LogisticRegression(random_state=40, solver='lbfgs', max_iter=10000, C=100, penalty='l2')
}

# Iterate over classifiers
for name, clf in classifiers.items():
    start_time = time.time()

    print(f"\n{name}:")
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    end_time = time.time()

    # Evaluate the classifier
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, average='macro') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='macro') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='macro') * 100, 2)
    
    # Print results
    print(f'Accuracy: {accuracy}% \n Precision: {precision}% \n Recall: {recall}% \n F1 Score: {f1}%')
    execution_time = end_time - start_time
    print(f"Execution time: {round(execution_time, 2)} s")