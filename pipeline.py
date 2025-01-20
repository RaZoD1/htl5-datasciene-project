import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('Train_data.csv')

# Drop unnecessary columns
columns_to_drop = ['num_outbound_cmds', 'is_host_login']
df = df.drop(columns=columns_to_drop)

# Prepare the feature matrix X and target vector y
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline for numerical data (impute and scale)
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler())  # Scale numerical data
])

# Preprocessing pipeline for categorical data (impute and one-hot encode)
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
])

# Combine both pipelines into a single ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Create the pipeline for decision tree model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 1. Cross-validation
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

# 2. Grid Search for hyperparameter tuning (using DecisionTree as an example)
param_grid = {
    'classifier__max_depth': [3, 5, 10, 15 ,20, 25, None],  # Max depth of the tree
    'classifier__min_samples_split': [2, 10, 20],  # Min samples to split a node
    'classifier__min_samples_leaf': [1, 5, 10]  # Min samples to be a leaf node
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f'Best parameters from grid search: {grid_search.best_params_}')
print(f'Best cross-validation score from grid search: {grid_search.best_score_}')

# 3. Evaluation on test set with the best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))



import matplotlib.pyplot as plt
import numpy as np

# Plot feature importances
def plot_feature_importance(model, X_train):
    # Get feature importance from the model
    importances = model.named_steps['classifier'].feature_importances_
    
    # Get the column names for the features
    feature_names = numerical_cols.tolist() + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols))
    
    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Plotting with a taller figure
    plt.figure(figsize=(10, 20))  # Increase the height of the figure
    plt.title("Feature Importances from the Decision Tree Classifier")
    plt.barh(range(len(importances)), importances[indices], align="center")
    plt.yticks(range(len(importances)), np.array(feature_names)[indices], fontsize=10)
    plt.xlabel("Importance")
    
    # Rotate y-axis labels and increase spacing
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Plot the feature importances for the best model (Decision Tree)
plot_feature_importance(best_model, X_train)



from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Predict with the best model from grid search
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
