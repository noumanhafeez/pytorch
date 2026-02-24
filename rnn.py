import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import re

# ------------------------------
# 1. Hyperparameters
# ------------------------------
MAX_VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 2. Text Cleaning
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ------------------------------
# 3. Build Vocabulary
# ------------------------------
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)

    return vocab

def encode_text(text, vocab):
    tokens = text.split()
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return encoded[:MAX_LEN] + [0] * max(0, MAX_LEN - len(encoded))

# ------------------------------
# 4. Dataset Class
# ------------------------------
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(self.texts[idx], self.vocab)
        return torch.tensor(encoded), torch.tensor(self.labels[idx], dtype=torch.float32)

# ------------------------------
# 5. LSTM Model
# ------------------------------
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# ------------------------------
# 6. Load Data
# ------------------------------
data = pd.read_csv("reviews.csv")
data["review"] = data["review"].astype(str).apply(clean_text)

texts = data["review"].tolist()
labels = data["sentiment"].values

vocab = build_vocab(texts)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_dataset = ReviewDataset(X_train, y_train, vocab)
test_dataset = ReviewDataset(X_test, y_test, vocab)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ------------------------------
# 7. Initialize Model
# ------------------------------
model = SentimentRNN(len(vocab)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------------
# 8. Training Loop
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# ------------------------------
# 9. Evaluation
# ------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).squeeze()
        preds = (outputs > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

print("\nAccuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# ------------------------------
# 10. Predict New Review
# ------------------------------
def predict_review(review):
    model.eval()
    review = clean_text(review)
    encoded = encode_text(review, vocab)
    tensor = torch.tensor([encoded]).to(device)

    with torch.no_grad():
        output = model(tensor).item()

    return "Positive 😊" if output > 0.5 else "Negative 😡"


# Example
print("\nNew Prediction:")
print(predict_review("This product is absolutely amazing and high quality"))