import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load features and labels
features_dir = '/home/melan/supervised-vs-SSL/data/results/passt/final_embeddings'
labels_csv = '/home/melan/supervised-vs-SSL/data/labels_mapping.csv'

embedding_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
labels_df = pd.read_csv(labels_csv)

X = []
y = []
for emb_file in embedding_files:
    emb_path = os.path.join(features_dir, emb_file)
    label_row = labels_df[labels_df['embedding_file'] == emb_file]
    if label_row.empty:
        continue  # skip if no label
    label = label_row['label'].values[0]
    if label == '' or label == 'unknown':
        continue  # skip if no valid label
    X.append(np.load(emb_path).flatten())
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")