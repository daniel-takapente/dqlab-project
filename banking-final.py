import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt

# Load Data
url = 'bank-additional-full.csv'
data = pd.read_csv(url, sep=',')

# Clean column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Check for and handle missing values
data = data.dropna()

# Define the columns to use (including target column)
selected_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'duration','poutcome',
                    'target']

# Select only specific columns from the data
data = data[selected_columns]

# Encode categorical variables
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']

# Check for missing columns before encoding
missing_cats = set(categorical_features) - set(data.columns)
if missing_cats:
    print(f"Warning: Missing categorical columns: {missing_cats}")
"""
# Plot frequency of each category for the existing categorical columns
for feature in categorical_features:
    if feature in data.columns:
        plt.figure(figsize=(10, 6))
        data[feature].value_counts().plot(kind='bar')
        plt.title(f'Frequency of Categories in {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
        plt.show()
"""
# Perform one-hot encoding only on existing columns
data = pd.get_dummies(data, columns=[col for col in categorical_features if col in data.columns], drop_first=True)

# Handle target variable ('y' column)
if 'target' in data.columns:
    label_encoder = LabelEncoder()
    data['target'] = label_encoder.fit_transform(data['target'])
    #data = data.drop('y', axis=1)  # Drop the original 'y' column after encoding
else:
    print("Error: Target column 'y' not found. Please check your dataset.")
    exit() # Stop execution if 'y' is missing

# Split data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
# Choosing the number of components to explain ~95% of variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
"""
# Plot the first two principal components
plt.figure(figsize=(10, 7))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.title('PCA of Bank Marketing Dataset (Training Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target')
plt.grid(True)
plt.show()
"""
# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42)
}

# Train, predict, and evaluate each model
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Collect accuracy, precision, recall, and f1-score
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print the comparison table
print("\nComparison of Model Performance:")
print(results_df)

# Save the Random Forest model and scaler (optional)
joblib.dump(models["Gradient Boosting"], 'bank_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModels have been evaluated and Gradient Boosting model and scaler saved successfully!")
