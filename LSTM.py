import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('SC_4label.csv')

# Tokenize the code
import nltk
nltk.download('punkt')
df['code_tokens'] = df['code'].apply(word_tokenize)

# Train Word2Vec model
word2vec_model_code = Word2Vec(df['code_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Define function to get word embeddings
def get_word_embeddings(tokens, word2vec_model):
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[token])
    return embeddings

# Apply word embeddings to the code tokens
df['code_embeddings'] = df['code_tokens'].apply(lambda x: get_word_embeddings(x, word2vec_model_code))

# Define sequence length
MAX_LENGTH = 100

# Pad sequences to ensure uniform length
X_pad = pad_sequences(df['code_embeddings'], maxlen=MAX_LENGTH, dtype='float32', padding='post')

# Scale the embeddings
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_pad.reshape(-1, MAX_LENGTH * 100))

# Assuming you have a target variable 'label_encoded' in your DataFrame
y = df['label_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(word2vec_model_code.wv.index_to_key)+1, output_dim=100, input_length=MAX_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
