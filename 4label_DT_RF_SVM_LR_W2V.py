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

# Define function to get word embeddings
def get_word_embeddings(tokens, word2vec_model):
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:  # Check if token is in the vocabulary
            embeddings.append(word2vec_model.wv[token])
    return embeddings


# Define the parameters for grid search
params = {
    "vector_size": [200, 300],
    "window": [7, 15],
    "min_count": [5, 10],
    "workers": [5, 10],
}

# Initialize the best parameters
best_params = {}
best_score = 0

# Grid search
for vector_size in params["vector_size"]:
    for window in params["window"]:
        for min_count in params["min_count"]:
            for workers in params["workers"]:
                                        # Train Word2Vec model with the current parameters
                                        word2vec_model_code = Word2Vec(df['code_tokens'], vector_size=vector_size, window=window, min_count=min_count, workers=workers, seed=42)
                                        print("Hi, I'm still running, dw")
                                        # Apply word embeddings to the code tokens, iter=iter
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

                                        # Train the classifier
                                        clf = RandomForestClassifier(random_state=40, max_depth=20, min_samples_split=2, n_estimators=200)
                                        clf.fit(X_train, y_train)

                                        # Predict on test set
                                        y_pred = clf.predict(X_test)

                                        # Evaluate the classifier
                                        accuracy = accuracy_score(y_test, y_pred)

                                        # Update the best parameters if the current accuracy is better than the best accuracy
                                        if accuracy > best_score:
                                            best_score = accuracy
                                            best_params = {"vector_size": vector_size, "window": window, "min_count": min_count, "workers": workers}

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_score}")

