import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# Load the data
data = pd.read_csv("preds.csv")

# Function to calculate and print evaluation metrics
def print_evaluation_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))
    
    return accuracy, precision, recall, f1

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

# Ground truth
y_true = data['folder']

# Predictions from different models with new names
pred_cols = {
    'poem_pred': 'poem_3d_finetuned',
    'poem_orig_pred': 'poem',
    'comb_pred': 'interjoint_distances'
}

# DataFrame to store metrics
metrics_list = []

# Evaluate each model
for col, model_name in pred_cols.items():
    y_pred = data[col]
    accuracy, precision, recall, f1 = print_evaluation_metrics(y_true, y_pred, model_name)
    metrics_list.append({'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
    plot_confusion_matrix(y_true, y_pred, model_name)

# Create metrics DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Display metrics table
print(metrics_df)

# Plotting the metrics
metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 6))
plt.title('Evaluation Metrics for Different Models')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend(loc='lower right')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
