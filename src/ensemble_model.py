"""
QSAR Biodegradability Prediction - Ensemble Methods
Author: Awon Alnsour
Course: Data Modelling and Machine Intelligence
University of Sheffield

This script implements ensemble learning methods for QSAR biodegradability prediction.
It builds on predictions from individual classifiers (RF, SVM, Custom MLP) to improve
overall performance. The ensembles implemented include:

1. Hard Voting: Majority vote of base model predictions
2. Weighted Voting: Weighted average of probabilities based on individual model accuracy
3. Stacking: Logistic Regression meta-classifier combining base model probabilities

Each ensemble returns comprehensive performance metrics including accuracy, precision, 
recall, F1 score, and additional details for stacking such as confusion matrix and 
sensitivity/specificity.

Dependencies:
- numpy
- scikit-learn
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def create_voting_ensembles(results, X_test, y_test):
    """
    Create hard and weighted voting ensembles from the base models.

    Hard Voting: Predicts the class based on majority vote from base models.
    Weighted Voting: Combines predicted probabilities weighted by model accuracy.

    Parameters:
    results : dict
        Dictionary containing predictions and probabilities from individual models.
    X_test : array-like
        Test features (not used directly, but passed for signature consistency).
    y_test : array-like
        True labels for test set.

    Returns:
    ensemble_results : dict
        Dictionary containing predictions and metrics for both ensembles.
    """
    print("\nIII) ENSEMBLE METHODS")
    
    ensemble_results = {}
    
    # Compute hard voting predictions by majority vote
    print("\n1. Hard Voting Ensemble...")
    rf_pred = results['RF']['predictions']
    svm_pred = results['SVM']['predictions']
    mlp_pred = results['MLP']['predictions']
    
    # Stack predictions and compute mean vote
    hard_votes = np.array([rf_pred, svm_pred, mlp_pred])
    hard_pred = np.round(np.mean(hard_votes, axis=0)).astype(int)
    
    # Store metrics for hard voting ensemble
    ensemble_results['Hard'] = {
        'predictions': hard_pred,
        'accuracy': accuracy_score(y_test, hard_pred),
        'precision': precision_score(y_test, hard_pred, zero_division=0),
        'recall': recall_score(y_test, hard_pred, zero_division=0),
        'f1': f1_score(y_test, hard_pred, zero_division=0)
    }
    
    # Compute weighted voting predictions based on model accuracies
    print("2. Weighted Voting Ensemble...")
    rf_acc = results['RF']['accuracy']
    svm_acc = results['SVM']['accuracy']
    mlp_acc = results['MLP']['accuracy']
    
    # Normalize accuracies to get weights
    weights = np.array([rf_acc, svm_acc, mlp_acc])
    weights = weights / weights.sum()
    
    # Combine predicted probabilities using weights
    rf_proba = results['RF']['proba']
    svm_proba = results['SVM']['proba']
    mlp_proba = results['MLP']['proba']
    weighted_proba = (weights[0] * rf_proba +
                      weights[1] * svm_proba +
                      weights[2] * mlp_proba)
    
    weighted_pred = (weighted_proba >= 0.5).astype(int)
    
    # Store metrics for weighted voting ensemble
    ensemble_results['Weighted'] = {
        'predictions': weighted_pred,
        'accuracy': accuracy_score(y_test, weighted_pred),
        'precision': precision_score(y_test, weighted_pred, zero_division=0),
        'recall': recall_score(y_test, weighted_pred, zero_division=0),
        'f1': f1_score(y_test, weighted_pred, zero_division=0),
        'weights': weights
    }
    
    print(f"   Model weights: RF={weights[0]:.3f}, SVM={weights[1]:.3f}, MLP={weights[2]:.3f}")
    
    return ensemble_results


def create_stacking_ensemble(models, X_train, y_train, X_test, y_test):
    """
    Create a stacking ensemble using logistic regression as a meta-classifier.

    The meta-features are the predicted probabilities from the base models.
    The meta-classifier learns to combine these predictions to improve overall performance.

    Parameters:
    models : dict
        Dictionary of trained base models.
    X_train, y_train : array-like
        Training data and labels.
    X_test, y_test : array-like
        Test data and labels.

    Returns:
    stacking_results : dict
        Dictionary containing predictions, metrics, confusion matrix, and meta-classifier.
    """
    print("3. Stacking Ensemble (Logistic Regression meta-classifier)...")
    
    # Prepare meta-features from base model probabilities
    X_meta_train = np.column_stack([
        models['RF'].predict_proba(X_train)[:, 1],
        models['SVM'].predict_proba(X_train)[:, 1],
        models['MLP'].predict_proba(X_train)
    ])
    
    X_meta_test = np.column_stack([
        models['RF'].predict_proba(X_test)[:, 1],
        models['SVM'].predict_proba(X_test)[:, 1],
        models['MLP'].predict_proba(X_test)
    ])
    
    # Train logistic regression as meta-classifier
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(X_meta_train, y_train)
    
    # Make predictions with stacking ensemble
    stacking_pred = meta_clf.predict(X_meta_test)
    stacking_proba = meta_clf.predict_proba(X_meta_test)[:, 1]
    
    # Compute confusion matrix and extract TP, TN, FP, FN
    cm = confusion_matrix(y_test, stacking_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Store all relevant metrics for stacking ensemble
    stacking_results = {
        'predictions': stacking_pred,
        'proba': stacking_proba,
        'accuracy': accuracy_score(y_test, stacking_pred),
        'precision': precision_score(y_test, stacking_pred, zero_division=0),
        'recall': recall_score(y_test, stacking_pred, zero_division=0),
        'f1': f1_score(y_test, stacking_pred, zero_division=0),
        'auc': roc_auc_score(y_test, stacking_proba),
        'meta_clf': meta_clf,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    return stacking_results
