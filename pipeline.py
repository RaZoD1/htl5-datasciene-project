import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class DecisionTreeModel:
    def __init__(self, data_path, model_filename='model.pkl'):
        self.data_path = data_path
        self.model_filename = model_filename
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.preprocessor = None
        self.model_pipeline = None

    def load_data(self):
        # Load the dataset from the provided CSV URL
        df = pd.read_csv(self.data_path)

        # Drop unnecessary columns
        columns_to_drop = ['num_outbound_cmds', 'is_host_login']
        df = df.drop(columns=columns_to_drop)

        # Prepare the feature matrix X and target vector y
        X = df.drop('class', axis=1)  # Features
        y = df['class']  # Target

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def preprocess_data(self):
        # Define numerical and categorical columns
        numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.X_train.select_dtypes(include=['object']).columns

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
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        # Fit the preprocessor on the training data
        self.preprocessor.fit(self.X_train)

    def create_model_pipeline(self):
        # Create the pipeline for Random Forest model
        self.model_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

    def cross_validate(self):
        # 1. Cross-validation
        cv_scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=5)
        print(f'Cross-validation scores: {cv_scores}')
        print(f'Mean cross-validation score: {cv_scores.mean()}')

    def grid_search(self):
        # 2. Grid Search for hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(self.model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        # Best parameters and best score
        print(f'Best parameters from grid search: {grid_search.best_params_}')
        print(f'Best cross-validation score from grid search: {grid_search.best_score_}')

        # Set the best model
        self.best_model = grid_search.best_estimator_

    def evaluate_model(self):
        # 3. Evaluation on test set with the best model from grid search
        y_pred = self.best_model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))

    def plot_feature_importance(self):
        # Plot feature importances
        importances = self.best_model.named_steps['classifier'].feature_importances_

        # Get the column names for the features
        feature_names = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist() + \
                        list(self.best_model.named_steps['preprocessor'].transformers_[1][1].named_steps[
                                 'onehot'].get_feature_names_out(
                            self.X_train.select_dtypes(include=['object']).columns))

        # Sort the feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Plotting with a taller figure
        plt.figure(figsize=(10, 20))  # Increase the height of the figure
        plt.title("Feature Importances from the Random Forest Classifier")
        plt.barh(range(len(importances)), importances[indices], align="center")
        plt.yticks(range(len(importances)), np.array(feature_names)[indices], fontsize=10)
        plt.xlabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        # Plot confusion matrix
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def save_model(self):
        # Save the trained model to a file
        joblib.dump(self.best_model, self.model_filename)
        print(f'Model saved as {self.model_filename}')

    def load_model(self):
        # Load the model from a file if it exists
        try:
            self.best_model = joblib.load(self.model_filename)
            print(f'Model loaded from {self.model_filename}')
        except FileNotFoundError:
            print(f'No model found at {self.model_filename}, proceeding with training.')

    def predict(self, X_new):
        # Make predictions for new data
        if self.best_model is None:
            print("Model is not trained yet!")
            return None

        # Predict and return results
        predictions = self.best_model.predict(X_new)
        return predictions

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.create_model_pipeline()

        # Load existing model or train a new one
        self.load_model()
        if not self.best_model:
            self.cross_validate()
            self.grid_search()
            self.evaluate_model()
            self.plot_feature_importance()
            self.plot_confusion_matrix()
            self.save_model()  # Save the trained model


DATA_PATH="https://docs.google.com/spreadsheets/d/e/2PACX-1vQkqK3rzUUOf-RIkiSU5RszMzHVwYgPTJUek6qjDrW6_F3MyJ-eETUa5UgiRzNdt6PhFtcKI6gioaj6/pub?gid=1746802197&single=true&output=csv"

if __name__ == '__main__':
    model = DecisionTreeModel(DATA_PATH)
    model.run()

  

    print(model.predict(model.X_test))
