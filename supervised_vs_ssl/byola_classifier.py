import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

features_dir = '/home/melan/supervised-vs-SSL/data/results/byola/final_embeddings_12k'
labels_csv = '/home/melan/supervised-vs-SSL/data/byola_labels_mapping.csv'

embedding_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])

labels_df = pd.read_csv(labels_csv)
labels_df['label'] = labels_df['label'].fillna('unknown').astype(str)
labels_map = {os.path.splitext(os.path.basename(f))[0]: str(label) for f, label in zip(labels_df['file_path'], labels_df['label'])}

X = []
y = []
print("Loading embeddings and labels...")
for i, fname in enumerate(embedding_files):
    emb = np.load(os.path.join(features_dir, fname))
    basename = os.path.splitext(fname)[0]
    label = labels_map.get(basename, 'unknown')
    X.append(emb.flatten())
    y.append(label)
    if i % 500 == 0 or i == len(embedding_files) - 1:
        print(f"Loaded {i+1}/{len(embedding_files)} files...")

X = np.stack(X)
y = np.array(y)

top_classes = [label for label, count in Counter(y).most_common(10)]
mask = np.isin(y, top_classes)
X = X[mask]
y = y[mask]

print(f"Filtered to top 10 classes: {top_classes}")
print(f"Final dataset: {X.shape[0]} samples, {len(top_classes)} classes.")

print("Training MLPClassifier...")
clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, random_state=42, verbose=True)
clf.fit(X, y)
print("Training complete.")

print("Predicting...")
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Accuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred, labels=top_classes))
print("\nClassification Report:")
print(classification_report(y, y_pred, labels=top_classes))

import csv
output_csv = 'byola_predictions.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'actual_label', 'predicted_label'])
    for fname, actual, pred in zip(embedding_files, y, y_pred):
        writer.writerow([fname, actual, pred])

print(f"Saved predictions to {output_csv}")