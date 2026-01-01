"""
QSAR Biodegradability Prediction - Utility Functions
Author: Awon Alnsour
Course: Data Modelling and Machine Intelligence
University of Sheffield

This script provides helper functions for visualization and result reporting.

Functions included:
1. generate_visualizations: Generate ROC curves, confusion matrix, feature importance, performance comparison, class distribution, and precision-recall curves.
2. print_results: Print formatted tables for individual classifiers, ensemble comparison, stacking ensemble performance, and feature importance.

Dependencies:
- numpy, matplotlib
- scikit-learn
"""

import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os


def generate_visualizations(results, ensemble_results, stacking_results, 
                           selector, X_train, y_train, X_test, y_test, y_original, FIGURES_DIR):
    """
    Generate all visualization figures for the report.

    Produces six key figures:
    1. ROC curves for all classifiers and stacking ensemble
    2. Confusion matrix for stacking ensemble
    3. Feature importance bar chart (top 10 features)
    4. Ensemble and individual model performance comparison
    5. Class distribution before and after SMOTE
    6. Precision-Recall curves

    Figures are saved as PDFs in the specified directory.
    """
    print("\nIV) GENERATING VISUALIZATIONS")
    
    # Create figures directory if missing
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # ROC Curves
    print("\n1. Generating ROC curves comparison...")
    plt.figure(figsize=(8, 6))
    all_results = results.copy()
    all_results['Stacking'] = stacking_results
    for name in ['RF', 'SVM', 'MLP', 'Stacking']:
        fpr, tpr, _ = roc_curve(y_test, all_results[name]['proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Classifier Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion matrix for stacking ensemble
    print("2. Generating confusion matrix for stacking ensemble...")
    cm = stacking_results['confusion_matrix']
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    classes = ['Non-biodegradable', 'Biodegradable']
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Stacking Ensemble Confusion Matrix')
    ax.set_ylabel('True Label', labelpad=15)
    ax.set_xlabel('Predicted Label', labelpad=15)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center")
    ax.tick_params(axis='y', pad=15)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance
    print("3. Generating feature importance bar chart...")
    if selector:
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        selected_scores = scores[selected_indices]
        top_10_idx = np.argsort(selected_scores)[-10:][::-1]
        features = [f'F{selected_indices[idx]+1}' for idx in top_10_idx]
        top_scores = selected_scores[top_10_idx]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, top_scores, color='steelblue', edgecolor='black')
        ax.set_xlabel('Mutual Information Score')
        ax.set_title('Top 10 Features by Mutual Information')
        ax.invert_yaxis()
        for bar, score in zip(bars, top_scores):
            ax.text(score + 0.001, bar.get_y() + bar.get_height()/2, f'{score:.4f}', va='center')
        ax.xaxis.grid(True, alpha=0.01, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    # Ensemble performance comparison
    print("4. Generating ensemble performance comparison bar chart...")
    methods = ['RF', 'SVM', 'MLP', 'Hard Voting', 'Weighted Voting', 'Stacking']
    accuracies = [results['RF']['accuracy'], results['SVM']['accuracy'], results['MLP']['accuracy'],
                  ensemble_results['Hard']['accuracy'], ensemble_results['Weighted']['accuracy'],
                  stacking_results['accuracy']]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['crimson', 'salmon', 'orange', 'gold', 'greenyellow', 'limegreen']
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim([0.8, 0.9])
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{acc:.3f}', ha='center')
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Class distribution before and after SMOTE
    print("5. Generating class distribution visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    original_counts = np.bincount(y_original)
    bars1 = ax1.bar(['Non-biodegradable', 'Biodegradable'], original_counts, color=['skyblue', 'lightcoral'], edgecolor='black')
    ax1.set_title('Original Class Distribution')
    ax1.set_ylabel('Count')
    ax1.set_ylim([0, max(original_counts) * 1.1])
    for bar, count in zip(bars1, original_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(original_counts)*0.02, str(count), ha='center')
    train_counts = np.bincount(y_train)
    bars2 = ax2.bar(['Non-biodegradable', 'Biodegradable'], train_counts, color=['skyblue', 'lightcoral'], edgecolor='black')
    ax2.set_title('Training Set After SMOTE')
    ax2.set_ylabel('Count')
    ax2.set_ylim([0, max(train_counts) * 1.1])
    for bar, count in zip(bars2, train_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_counts)*0.02, str(count), ha='center')
    for ax_sub in [ax1, ax2]:
        ax_sub.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax_sub.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Precision-Recall curves
    print("6. Generating precision-recall curves...")
    plt.figure(figsize=(8, 6))
    for name in ['RF', 'SVM', 'MLP', 'Stacking']:
        precision, recall, _ = precision_recall_curve(y_test, all_results[name]['proba'])
        plt.plot(recall, precision, label=name, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'precision_recall_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n All visualizations saved to '{FIGURES_DIR}'")


def print_results(results, ensemble_results, stacking_results, selector, y_test):
    """
    Print formatted performance tables for report.

    Sections include:
    1. Individual classifier performance
    2. Ensemble comparison
    3. Detailed stacking ensemble analysis
    4. Feature importance (top 10 features)
    """
    print("\nV) RESULTS")

    # Individual classifier metrics
    print("\nIndividual Classifier Performance")
    print(f"{'Classifier':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    for name in ['RF', 'SVM', 'MLP']:
        r = results[name]
        print(f"{name:<12} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f}")

    # Ensemble comparison
    print("\nEnsemble Method Comparison")
    print(f"{'Method':<20} {'Accuracy':<10}")
    print(f"{'Hard Voting':<20} {ensemble_results['Hard']['accuracy']:<10.4f}")
    print(f"{'Weighted Voting':<20} {ensemble_results['Weighted']['accuracy']:<10.4f}")
    print(f"{'Stacking':<20} {stacking_results['accuracy']:<10.4f}")

    # Stacking ensemble details
    print("\nStacking Ensemble Detailed Performance")
    stacking_pred = stacking_results['predictions']
    print(classification_report(y_test, stacking_pred, target_names=['Non-biodegradable', 'Biodegradable']))
    print(f"Confusion Matrix: TN={stacking_results['tn']}, FP={stacking_results['fp']}, FN={stacking_results['fn']}, TP={stacking_results['tp']}")
    print(f"Sensitivity (TPR): {stacking_results['sensitivity']:.3f}")
    print(f"Specificity (TNR): {stacking_results['specificity']:.3f}")

    # Feature importance
    if selector:
        print("\nTop 10 Selected Features by Mutual Information:")
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        selected_scores = scores[selected_indices]
        top_10_idx = np.argsort(selected_scores)[-10:][::-1]
        for i, idx in enumerate(top_10_idx):
            feat_idx = selected_indices[idx]
            score = selected_scores[idx]
            print(f"  {i+1:2d}. Feature {feat_idx+1:3d}: {score:.4f}")
