import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np


def set_random_seeds(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking for reproducibility

# Set random seeds
set_random_seeds()

# Now continue with your training and evaluation code

class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='linear', dropout=0.0):
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask=None):
        '''x: (B, T, D)'''
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        kvw = k * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out = kvw.sum(dim=-2, keepdim=True) * q
        return self.out(out)
    




# Assuming you've defined the HydraAttention class already

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
X = torch.tensor(code_embeddings_scaled).float()
y = torch.tensor(df['label_encoded'].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize and train the HydraAttention model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HydraAttention(d_model=100, output_layer='linear', dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the trained model
with torch.no_grad():
    inputs_test, labels_test = X_test.to(device), y_test.to(device)
    outputs_test = model(inputs_test)
    _, predicted = torch.max(outputs_test, 1)
    accuracy = accuracy_score(labels_test.cpu().numpy(), predicted.cpu().numpy())
    precision = precision_score(labels_test.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    recall = recall_score(labels_test.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    f1 = f1_score(labels_test.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")