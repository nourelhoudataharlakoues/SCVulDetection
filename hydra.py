import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler

# Assuming you have already defined the HydraAttention class
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

# Compute average embeddings for each code snippet
df['code_avg_embedding'] = df['code_embeddings'].apply(lambda x: sum(x) / len(x))

# Convert embeddings to tensor
X = torch.tensor(df['code_avg_embedding'].to_list(), dtype=torch.float32)

# Assuming you have a target variable 'label_encoded' in your DataFrame
y = torch.tensor(df['label_encoded'].values, dtype=torch.long)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define HydraAttention model
class HydraAttentionClassifier(nn.Module):
    def __init__(self, input_size, num_classes, d_model, output_layer='linear', dropout=0.0):
        super(HydraAttentionClassifier, self).__init__()
        self.attention = HydraAttention(d_model, output_layer=output_layer, dropout=dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        attention_out = self.attention(x)
        logits = self.fc(attention_out.squeeze())
        return logits

# Initialize and train the model
model = HydraAttentionClassifier(input_size=X_train.size(1), num_classes=4, d_model=100)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    precision = precision_score(y_test.numpy(), predicted.numpy(), average='macro')
    recall = recall_score(y_test.numpy(), predicted.numpy(), average='macro')
    f1 = f1_score(y_test.numpy(), predicted.numpy(), average='macro')

# Print results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
