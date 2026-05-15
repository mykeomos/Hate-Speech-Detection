# Hate Speech Detection

## Overview

This repository contains a Jupyter notebook for hate speech detection using classical machine learning models. The notebook performs end-to-end data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and a simple deployment simulation.

## Notebook

- `Hate Speech Detection.ipynb`: Main notebook implementing the workflow.

## Dataset

- `labeled_data.csv`: Labeled tweet dataset used for training and evaluation.

> Note: The notebook expects `labeled_data.csv` to be available in the same working directory when executed. If you open the notebook from `Hate-Speech-Detection/`, you may need to copy or link the dataset into that folder or adjust the file path accordingly.

## Key Steps Covered

1. Data exploration and preprocessing
   - Load dataset
   - Inspect structure, missing values, and class distribution
   - Clean tweet text using lowercase conversion, URL removal, mention removal, punctuation stripping, and whitespace trimming

2. Feature engineering
   - Extract text features using TF-IDF
   - Reduce dimensionality with PCA while preserving most variance

3. Model training
   - Train multiple classifiers:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - Gradient Boosting
     - Multinomial Naive Bayes

4. Hyperparameter tuning
   - Use `GridSearchCV` to tune each model with weighted F1 scoring

5. Evaluation and comparison
   - Compute accuracy, precision, recall, F1-score, confusion matrices
   - Generate classification reports
   - Compare model performance in a summary table
   - Select the best-performing model

6. Deployment simulation
   - Define a prediction function
   - Demonstrate model inference on unseen text input

## How to Run

1. Open `Hate Speech Detection.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure Python libraries are installed:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
3. Place `labeled_data.csv` in the notebook's working directory or update the path in the notebook.
4. Run cells sequentially from top to bottom.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Notes

- The notebook contains two TF-IDF feature pipelines: one for PCA-based training and one for Naive Bayes using raw TF-IDF features.
- Model fairness is partially addressed using `class_weight='balanced'` for several classifiers.
- The best model is selected based on evaluation metrics and demonstrated with a prediction helper function.
