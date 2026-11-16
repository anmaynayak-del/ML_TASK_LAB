import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Open file with pd.read_csv ('data.csv' or 'Iris.csv')
df = pd.read_csv('data.csv')
print("Shape of dataset (rows, columns):", df.shape)

# Print head of dataset
print("\n************** First 5 rows of dataset **************\n", df.head())

# Select features and target based on dataset
if 'Species' in df.columns:
    drop_cols = [col for col in ['Id', 'Species'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['Species']
    target_names = sorted(df['Species'].unique())
elif 'Outcome' in df.columns:
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    target_names = ['Normal', 'Diabetes']
elif 'diagnosis' in df.columns:
    drop_cols = [col for col in ['id', 'diagnosis', 'Unnamed: 32'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['diagnosis']
    target_names = ['Benign (B)', 'Malignant (M)']
else:
    raise ValueError('Target column not recognized in dataset')

print("\nFeatures set:\n", X.head())
print("\nTarget labels:\n", y.head())

# Implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random forest model creation
rfc = RandomForestClassifier(n_estimators=3, max_features=5, random_state=42)
rfc.fit(X_train, y_train)

# Predictions
rfc_train_predict = rfc.predict(X_train)
rfc_predict = rfc.predict(X_test)

# ==================== Accuracy Section ====================
train_accuracy = accuracy_score(y_train, rfc_train_predict)
test_accuracy = accuracy_score(y_test, rfc_predict)
correct_count = (y_test == rfc_predict).sum()
total_count = len(y_test)

print("\n" + "=" * 45)
print("             ACCURACY SECTION")
print("=" * 45)
print(f"Training Accuracy   : {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
print(f"Testing Accuracy    : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Correct Predictions : {correct_count} / {total_count}")
print(f"Misclassified       : {total_count - correct_count} / {total_count}")
print("=" * 45 + "\n")

# Evaluation
# 1. Confusion Matrix
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')

# 2. Classification Report
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict, target_names=target_names))

# Visualizing True vs Predicted values
index = np.arange(0, len(y_test))
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.scatter(index, y_test, c="red", label='True Value', alpha=0.7)
plt.scatter(index, rfc_predict, c="blue", label='Predicted Value', marker='x')
plt.title(f'Random Forest Classifier: True vs Predicted (Accuracy: {test_accuracy * 100:.2f}%)')
plt.xlabel('Sample Index')
plt.ylabel('Diagnosis')
plt.legend()

# Save plot to file
plt.savefig('true_vs_predicted.png', bbox_inches='tight')
print("Plot saved as 'true_vs_predicted.png'")

# Display the plot
if os.environ.get('NON_INTERACTIVE') == '1':
    plt.close()
else:
    plt.show()
