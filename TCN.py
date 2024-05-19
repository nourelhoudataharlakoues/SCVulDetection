# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import pandas as pd
# from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
# from sklearn.preprocessing import MinMaxScaler

# # Load the dataset
# df = pd.read_csv('SC_4label.csv')

# # Tokenize the code
# import nltk
# nltk.download('punkt')
# df['code_tokens'] = df['code'].apply(word_tokenize)

# # Train Word2Vec model
# word2vec_model_code = Word2Vec(df['code_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# # Define function to get word embeddings
# def get_word_embeddings(tokens, word2vec_model):
#     embeddings = []
#     for token in tokens:
#         if token in word2vec_model.wv:
#             embeddings.append(word2vec_model.wv[token])
#     return embeddings

# # Apply word embeddings to the code tokens
# df['code_embeddings'] = df['code_tokens'].apply(lambda x: get_word_embeddings(x, word2vec_model_code))

# # Compute average embeddings for each code snippet
# df['code_avg_embedding'] = df['code_embeddings'].apply(lambda x: sum(x) / len(x))

# # Convert embeddings to tensor
# X = torch.tensor(df['code_avg_embedding'].to_list(), dtype=torch.float32)

# # Assuming you have a target variable 'label_encoded' in your DataFrame
# y = torch.tensor(df['label_encoded'].values, dtype=torch.long)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define TCN model
# class TCNClassifier(nn.Module):
#     def __init__(self, input_size, num_channels, kernel_size, dropout):
#         super(TCNClassifier, self).__init__()
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
#         self.fc = nn.Linear(num_channels[-1], 4)  # Assuming 4 output classes

#     def forward(self, x):
#         y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
#         y = y[:, -1, :]  # Use the output at the last time step
#         return self.fc(y)

# # Define Temporal Convolutional Network (TCN) block
# class TemporalBlock(nn.Module):
#     def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(input_size, output_size, kernel_size,
#                                stride=stride, dilation=dilation, padding=padding)
#         self.chomp1 = nn.ConstantPad1d((0, 1), 0)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)

#     def forward(self, x):
#         out = self.conv1(self.chomp1(x))
#         out = self.relu1(out)
#         out = self.dropout1(out)
#         return out

# # Define Temporal Convolutional Network (TCN)
# class TemporalConvNet(nn.Module):
#     def __init__(self, input_size, num_channels, kernel_size, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = input_size if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout)]
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

# # Initialize and train the model
# model = TCNClassifier(input_size=X_train.size(2), num_channels=[64, 64, 64], kernel_size=2, dropout=0.2)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train.unsqueeze(2))
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     outputs = model(X_test.unsqueeze(2))
#     _, predicted = torch.max(outputs, 1)
#     accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
#     precision = precision_score(y_test.numpy(), predicted.numpy(), average='macro')
#     recall = recall_score(y_test.numpy(), predicted.numpy(), average='macro')
#     f1 = f1_score(y_test.numpy(), predicted.numpy(), average='macro')

# # Print results
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

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
df['code_avg_embedding'] = df['code_embeddings'].apply(lambda x: np.mean(x, axis=0) if len(x) > 0 else np.zeros(100))

# Convert embeddings to tensor
X = np.array(df['code_avg_embedding'].tolist())
X = torch.tensor(X, dtype=torch.float32)

# Assuming you have a target variable 'label_encoded' in your DataFrame
y = torch.tensor(df['label_encoded'].values, dtype=torch.long)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test to have three dimensions
X_train = X_train.unsqueeze(2)
X_test = X_test.unsqueeze(2)

# Define TCN model
class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], 4)  # Assuming 4 output classes

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        y = y[:, -1, :]  # Use the output at the last time step
        return self.fc(y)

# Define Temporal Convolutional Network (TCN) block
class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.chomp1 = nn.ConstantPad1d((0, 1), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(self.chomp1(x))
        out = self.relu1(out)
        out = self.dropout1(out)
        return out

# Define Temporal Convolutional Network (TCN)
class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize and train the model
model = TCNClassifier(input_size=1, num_channels=[64, 64, 64], kernel_size=2, dropout=0.2)
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
