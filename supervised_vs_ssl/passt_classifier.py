import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
from sklearn.neural_network import MLPClassifier
import csv

features_dir = '/home/melan/supervised-vs-SSL/data/results/passt/final_embeddings'
labels_csv = '/home/melan/supervised-vs-SSL/data/passt_labels_mapping.csv'

embedding_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
print(f"Found {len(embedding_files)} embedding files in {features_dir}")

labels_df = pd.read_csv(labels_csv)
print(f"Loaded labels CSV with {len(labels_df)} rows and columns: {labels_df.columns.tolist()}")

X = []
y = []
missing_label_count = 0
unknown_label_count = 0
loaded_count = 0
file_names = []

print("Loading embeddings and labels...")
for i, emb_file in enumerate(embedding_files):
    emb_path = os.path.join(features_dir, emb_file)
    emb_base = os.path.splitext(emb_file)[0]
    label_row = labels_df[labels_df['file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) == emb_base]
    if label_row.empty:
        missing_label_count += 1
        if missing_label_count <= 5:
            print(f"Missing label for {emb_file}")
        continue
    label = label_row['label'].values[0]
    if label == '' or label == 'unknown':
        unknown_label_count += 1
        if unknown_label_count <= 5:
            print(f"Unknown/empty label for {emb_file}")
        continue
    try:
        emb = np.load(emb_path).flatten()
        X.append(emb)
        y.append(label)
        file_names.append(emb_file)
        loaded_count += 1
        if loaded_count % 500 == 0 or loaded_count <= 5:
            print(f"Loaded {loaded_count}/{len(embedding_files)}: {emb_file}, label: {label}, emb shape: {emb.shape}")
    except Exception as e:
        print(f"Error loading {emb_file}: {e}")

print(f"Total loaded: {loaded_count}")
print(f"Missing labels: {missing_label_count}")
print(f"Unknown/empty labels: {unknown_label_count}")

X = np.array(X)
y = np.array(y)
print(f"Final X shape: {X.shape}, y shape: {y.shape}")

if len(X) == 0 or len(y) == 0:
    print("ERROR: No data loaded for classification. Check label matching and file paths.")
    exit()

# Split train/test
X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
    X, y, file_names, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

print("Class distribution:", Counter(y_train))
print("Number of unique classes:", len(set(y_train)))
print("NaNs in X_train:", np.isnan(X_train).sum())
print("Infs in X_train:", np.isinf(X_train).sum())

clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, random_state=42, verbose=True)
print("Training MLPClassifier (cross-entropy loss)...")
clf.fit(X_train, y_train)
print("Training complete.")

print("Predicting...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Save predictions
output_csv = 'passt_predictions.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'actual_label', 'predicted_label'])
    for fname, actual, pred in zip(test_files, y_test, y_pred):
        writer.writerow([fname, actual, pred])
print(f"Saved predictions to {output_csv}")

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Script finished. If you see this, everything worked.")