# QSAR Biodegradability Classification

**Author:** Awon Alnsour    
**University:** University of Sheffield  

## About this Project

This project explores **QSAR-based biodegradability classification** using a combination of **traditional machine learning models** (Random Forest, SVM, MLP) and **advanced ensemble strategies**. It highlights how **stacking ensembles with a meta-learner** can intelligently combine models, improving generalization beyond any single classifier.  

The pipeline includes **data preprocessing, class balancing, feature selection, model training, ensemble creation, evaluation, and visualization**, making it a comprehensive example of applied machine learning for chemical property prediction.


## Project Overview

The main script, [`src/main_run.py`](src/main_run.py), orchestrates the full workflow:

1. Load and preprocess QSAR data (handling missing values, SMOTE balancing, feature selection, scaling)
2. Train individual classifiers:  
   - Random Forest  
   - Support Vector Machine (RBF kernel)  
   - Custom Multi-Layer Perceptron (MLP)  
3. Build ensemble models:  
   - Hard and weighted voting  
   - Stacking with a logistic regression meta-learner  
4. Evaluate model performance (accuracy, precision, recall, F1-score, ROC-AUC)  
5. Generate visualizations for analysis  

The pipeline illustrates how **ensemble methods**, particularly stacking with a meta-learner, can **learn when to trust each base model** and outperform any single classifier.

## Dependencies

- Python ≥ 3.9
- numpy, scipy
- scikit-learn
- imbalanced-learn
- matplotlib
- Custom `mlp_model.py`, `ensemble_model.py`, and `utils.py`

You can install the main dependencies via:

```bash
pip install numpy scipy scikit-learn imbalanced-learn matplotlib
```
## How to run

You can run this script by running the following command in the terminal

```bash
python src/main_run.py
```

## Folder Structure
QSAR-classification/
├── src/
│   ├── main_run.py          # Main execution script (this file)
│   ├── mlp_model.py         # Custom SimpleMLP implementation
│   ├── ensemble_model.py    # Voting and stacking ensemble implementations
│   └── utils.py             # Visualization and utility functions
├── data/
│   └── QSAR_data.mat        # Dataset (MATLAB format)
├── plots/                   # Generated visualizations (PDF format)
├── README.md                # This file
└── requirements.txt         # Python dependencies

