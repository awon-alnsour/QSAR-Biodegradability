"""
QSAR Biodegradability Prediction - Main Execution Script
Author: Awon Alnsour
Course: Data Modelling and Machine Intelligence
University of Sheffield

This script implements a complete machine learning pipeline for predicting
chemical biodegradability from QSAR data. The pipeline includes:
1. Data preprocessing (SMOTE balancing, feature selection, scaling)
2. Training of three individual classifiers (RF, SVM, Custom MLP)
3. Implementation of ensemble methods (Hard/Weighted Voting, Stacking)
4. Comprehensive performance evaluation and visualization

Dependencies:
- numpy, scipy
- scikit-learn, imbalanced-learn
- matplotlib
- MLP_model (SimpleMLP implementation) see MLP_model.py
- os (for file and directory handling)
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import os

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# Import custom MLP model
from src.mlp_model import SimpleMLP

# Import ensemble and utility functions see utils.py, enesmble_model.py
from src.ensemble_model import create_stacking_ensemble, create_voting_ensembles
from src.utils import generate_visualizations, print_results

# Set random seed to make experiments reproducible
np.random.seed(42)

# Global configuration constants
DATA_FILE = os.path.join("data", "QSAR_data.mat")  # relative path to data folder
TEST_SIZE = 0.2
N_FEATURES_SELECTED = 25

# Figures directory (relative to project root, not src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "plots")

# folder exists
os.makedirs(FIGURES_DIR, exist_ok=True)


class SKLearnMLPWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for MLP to integrate with scikit-learn.
    This allows us to use the MLP alongside standard scikit-learn models
    and evaluate it with the same metrics.
    """
    
    def __init__(self, hidden_size=64, learning_rate=0.001, l2_reg=0.0, 
                 batch_size=32, dropout_prob=0.0, epochs=100, seed=42):
        """
        Initialize MLP wrapper with hyperparameters.

        Parameters:
        hidden_size : int
            Number of neurons in the hidden layer
        learning_rate : float
            Learning rate for training
        l2_reg : float
            L2 regularization strength
        batch_size : int
            Batch size during training
        dropout_prob : float
            Dropout probability for regularization
        epochs : int
            Number of training epochs
        seed : int
            Random seed for reproducibility
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.epochs = epochs
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        """
        Train the custom MLP model on the given data.

        Parameters:
        X : numpy array
            Feature matrix for training
        y : numpy array
            Target labels for training
        """
        # Initialize the SimpleMLP with the specified hyperparameters
        self.model_ = SimpleMLP(
            input_size=X.shape[1],
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            l2_reg=self.l2_reg,
            batch_size=self.batch_size,
            dropout_prob=self.dropout_prob,
            seed=self.seed
        )

        # Train the model
        self.model_.train(X, y, epochs=self.epochs, verbose=False)
        return self

    def predict(self, X):
        """Return class predictions for input data."""
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Return predicted probabilities for input data."""
        return self.model_.predict_proba(X)
    

def load_and_preprocess_data(DATA_FILE, N_FEATURES_SELECTED):
    """
    Load the QSAR dataset and preprocess it for modeling.

    Steps:
    1. Load data from MATLAB .mat file
    2. Check for missing values and impute if necessary
    3. Split into stratified train/test sets
    4. Balance classes using SMOTE on the training set
    5. Select top features using mutual information
    6. Standardize features

    Returns:
    X_train, X_test, y_train, y_test, selector, scaler, y_original
    """
    print("\nI) PREPROCESSING DATA")
    
    # Load MATLAB data
    mat_data = loadmat(DATA_FILE)
    data_array = mat_data['QSAR_data']
    X = data_array[:, :-1]
    y = data_array[:, -1].astype(int)
    
    # Basic dataset info
    print(f"\nDataset loaded: {data_array.shape}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Positive class ratio: {y.mean():.2%}")
    
    # Handle missing values if present
    nan_count = np.isnan(X).sum().sum()
    if nan_count > 0:
        print(f"Missing values found: {nan_count}, imputing median values")
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy='median').fit_transform(X)
    else:
        print("No missing values found")
    
    # Split into stratified train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train class distribution: {np.bincount(y_train)}")
    
    # Select top features using mutual information
    selector = SelectKBest(mutual_info_classif, k=N_FEATURES_SELECTED)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    print(f"\nSelected {N_FEATURES_SELECTED} features (from original {X.shape[1]})")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, selector, scaler, y


def train_individual_classifiers(X_train, y_train, X_test, y_test):
    """
    Train Random Forest, SVM, and custom MLP classifiers on the training data.

    Returns:
    models : dict of trained classifiers
    results : dict of performance metrics for each model
    """
    models = {}
    results = {}
    
    print("\nII) TRAINING INDIVIDUAL CLASSIFIERS")
    
    # Train Random Forest with 300 trees and max depth of 15
    print("\n1. Random Forest (n_estimators=300, max_depth=15)...")
    rf_model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    models['RF'] = rf_model
    results['RF'] = {
        'predictions': rf_pred,
        'proba': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
        'auc': roc_auc_score(y_test, rf_proba)
    }
    
    # Train Support Vector Machine with RBF kernel
    print("2. Support Vector Machine (C=10, RBF kernel)...")
    svm_model = SVC(
        C=10, 
        kernel='rbf', 
        gamma='scale',
        probability=True, 
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    
    models['SVM'] = svm_model
    results['SVM'] = {
        'predictions': svm_pred,
        'proba': svm_proba,
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred, zero_division=0),
        'recall': recall_score(y_test, svm_pred, zero_division=0),
        'f1': f1_score(y_test, svm_pred, zero_division=0),
        'auc': roc_auc_score(y_test, svm_proba)
    }
    
    # Train custom MLP model
    print("3. Custom MLP (256 hidden, dropout=0.1, lr=0.001)...")
    mlp_model = SKLearnMLPWrapper(
        hidden_size=256,
        learning_rate=0.001,
        dropout_prob=0.1,
        l2_reg=0.0,
        batch_size=32,
        epochs=300
    )
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_proba = mlp_model.predict_proba(X_test)
    
    models['MLP'] = mlp_model
    results['MLP'] = {
        'predictions': mlp_pred,
        'proba': mlp_proba,
        'accuracy': accuracy_score(y_test, mlp_pred),
        'precision': precision_score(y_test, mlp_pred, zero_division=0),
        'recall': recall_score(y_test, mlp_pred, zero_division=0),
        'f1': f1_score(y_test, mlp_pred, zero_division=0),
        'auc': roc_auc_score(y_test, mlp_proba)
    }
    
    return models, results


def main():
    """
    Orchestrate the complete QSAR prediction pipeline:
    1. Load and preprocess data
    2. Train individual classifiers
    3. Create ensemble models
    4. Generate visualizations
    5. Print results and summaries
    """
    print("Starting QSAR biodegradability prediction pipeline...")
    
    # Load and preprocess data, select features, and scale
    X_train, X_test, y_train, y_test, selector, scaler, y_original = load_and_preprocess_data(DATA_FILE, N_FEATURES_SELECTED)
    
    # Train RF, SVM, and MLP classifiers
    models, results = train_individual_classifiers(X_train, y_train, X_test, y_test)
    
    # Build hard and weighted voting ensembles
    ensemble_results = create_voting_ensembles(results, X_test, y_test)
    
    # Build stacking ensemble with logistic regression meta-classifier
    stacking_results = create_stacking_ensemble(models, X_train, y_train, X_test, y_test)
    
    # Generate all report visualizations and save figures
    generate_visualizations(results, ensemble_results, stacking_results, 
                           selector, X_train, y_train, X_test, y_test, y_original, FIGURES_DIR)
    
    # Print formatted results for all models
    print_results(results, ensemble_results, stacking_results, selector, y_test)
    
    # Print final summary and best model
    print("\nVI) SUMMARY\n")
    print(f"Best Model: Stacking Ensemble")
    print(f"Best Accuracy: {stacking_results['accuracy']:.4f}")
    
    # List all generated figures
    print(f"\nFigures saved in '{FIGURES_DIR}' directory:")
    for filename in sorted(os.listdir(FIGURES_DIR)):
        if filename.endswith('.pdf'):
            print(f"  - {filename}")
    
    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()
