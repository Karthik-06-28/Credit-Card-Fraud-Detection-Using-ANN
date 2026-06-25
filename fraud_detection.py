import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# distribution
def check_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{title}")
    for u, c in zip(unique, counts):
        print(f"Class {int(u)}: {c}")

    print("\n")

# load
train_df = pd.read_csv('/content/fraudTrain.csv')
test_df  = pd.read_csv('/content/fraudTest.csv')
test_display = test_df.copy()

# drop cols
drop_cols = [
    'Unnamed: 0','trans_date_trans_time','cc_num','merchant',
    'first','last','street','city','zip','lat','long',
    'dob','trans_num','unix_time','merch_lat','merch_long'
]
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# encode
df = pd.concat([train_df, test_df])

df = pd.concat([df, pd.get_dummies(df['category'], drop_first=True)], axis=1)
df.drop(columns=['category'], inplace=True)

df['gender'] = df['gender'].map({'F':1,'M':0})

le_state = LabelEncoder()
le_job = LabelEncoder()
df['state'] = le_state.fit_transform(df['state'])
df['job'] = le_job.fit_transform(df['job'])

# split
train_df = df.iloc[:len(train_df)]
test_df  = df.iloc[len(train_df):]

X_train = train_df.drop(columns=['is_fraud'])
y_train = train_df['is_fraud']

X_test = test_df.drop(columns=['is_fraud'])
y_test = test_df['is_fraud']

# before smote
check_distribution(y_train, "Before SMOTE")

# smote
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# after smote
check_distribution(y_train, "After SMOTE")

# scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

# model
class ANN(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.net(x)

model = ANN(X_train.shape[1])

# loss + optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# train
for epoch in range(12):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: {total_loss:.4f}")

# test
model.eval()
all_preds, all_probs, all_labels = [], [], []
THRESHOLD = 0.7

with torch.no_grad():
    for xb, yb in test_loader:
        probs = torch.sigmoid(model(xb))
        preds = (probs > THRESHOLD).float()
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())
        all_labels.extend(yb.numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# metrics
print("\nAccuracy:", accuracy_score(all_labels, all_preds))
print("ROC-AUC:", roc_auc_score(all_labels, all_probs))
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# confusion matrix
cf = confusion_matrix(all_labels, all_preds)
sns.heatmap(cf, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# sample predictions
print("\nSample Test Predictions:\n")

for i in range(10):
    print(f"Transaction {i+1}")
    print(f"Amount: {test_display.iloc[i]['amt']}")
    print(f"Category: {test_display.iloc[i]['category']}")
    print(f"Gender: {test_display.iloc[i]['gender']}")
    print(f"State: {test_display.iloc[i]['state']}")
    print(f"Job: {test_display.iloc[i]['job']}")
    print(f"Predicted Prob: {all_probs[i][0]:.4f}")
    print(f"Predicted Class: {int(all_preds[i][0])}")
    print(f"Actual Class: {int(all_labels[i][0])}")
    print("-"*40)