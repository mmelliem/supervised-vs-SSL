import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

features_dir = '/home/melan/supervised-vs-SSL/data/results/byola/final_embeddings'
labels_csv = '/home/melan/supervised-vs-SSL/data/byola_labels_mapping.csv'

# Load all embeddings
embedding_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
X = []
file_basenames = []
for i, emb_file in enumerate(embedding_files):
    emb_path = os.path.join(features_dir, emb_file)
    try:
        arr = np.load(emb_path)
        print(f"[{i+1}/{len(embedding_files)}] Loaded {emb_file}, shape: {arr.shape}")
        X.append(arr.flatten())
        file_basenames.append(os.path.splitext(emb_file)[0])
    except Exception as e:
        print(f"Error loading {emb_file}: {e}")
X = np.array(X)

# Load labels from CSV
labels_df = pd.read_csv(labels_csv)
labels_map = {os.path.splitext(os.path.basename(f))[0]: label for f, label in zip(labels_df['file_path'], labels_df['label'])}
y = np.array([labels_map.get(basename, 'unknown') for basename in file_basenames])
print("First 5 label mappings:", list(labels_map.items())[:5])

# Filter out unknown labels
print(f"Before filtering: X shape {X.shape}, y shape {y.shape}")
mask = y != 'unknown'
X = X[mask]
y = y[mask]
print(f"After filtering: X shape {X.shape}, y shape {y.shape}")

# Diagnostics
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("Class distribution:", Counter(y))
print("Number of unique classes:", len(set(y)))
print("NaNs in X:", np.isnan(X).sum())
print("Infs in X:", np.isinf(X).sum())

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ...existing code up to train/test split...

# Limit to top 10 most common classes
common_labels = [label for label, count in Counter(y_train).most_common(10)]
print("Top 10 labels:", common_labels)

mask_train = np.isin(y_train, common_labels)
mask_test = np.isin(y_test, common_labels)

X_train_small = X_train[mask_train]
y_train_small = y_train[mask_train]
X_test_small = X_test[mask_test]
y_test_small = y_test[mask_test]

print(f"Filtered train size: {len(X_train_small)}, test size: {len(X_test_small)}")
print("Filtered train class distribution:", Counter(y_train_small))

# Train SGDClassifier
clf = SGDClassifier(max_iter=100, random_state=42)
print("Training SGDClassifier on top 10 classes...")
clf.fit(X_train_small, y_train_small)

# Evaluate
y_pred = clf.predict(X_test_small)
acc = accuracy_score(y_test_small, y_pred)
print(f"Test accuracy (top 10 classes): {acc:.4f}")