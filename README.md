# Hate Speech Detection

## Overview
This repository contains a Jupyter notebook that implements a classical machine-learning pipeline for hate-speech detection on a labeled Twitter dataset. The notebook walks through data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and a minimal deployment example.

## Files
- `Hate Speech Detection.ipynb` — main analysis notebook (data loading, models, evaluation).
- `labeled_data.csv` — labeled tweets used as the dataset.

## Notebook Summary
The notebook is organized into the following sections:
- **1. Data Exploration & Pre-processing**: loads `labeled_data.csv`, inspects class distribution, checks missing values and samples, and applies a `clean_text` function (lowercasing, URL/mention/hashtag removal, punctuation stripping).
- **2. Feature Engineering**: uses `TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))` to convert cleaned text into TF-IDF features; then applies PCA to retain 95% variance for dimensionality reduction (converting the TF-IDF sparse matrix to dense before PCA).
- **3. Model Selection & Training**: trains multiple classifiers — Logistic Regression, SVM (linear), Random Forest, Gradient Boosting, and Multinomial Naïve Bayes (NB uses TF-IDF without PCA because NB expects non-negative features).
- **4. Hyperparameter Tuning**: performs GridSearchCV for each model (with `f1_weighted` scoring and 5-fold CV) to find best hyperparameters.
- **5. Evaluation & Comparative Analysis**: evaluates models using Accuracy, Precision, Recall, F1-score, ROC-AUC, Log Loss, and Confusion Matrices, then creates a comparison table and selects the best model by F1-Score.
- **6. Minimal Deployment**: provides a `predict_hate_speech` helper function to clean, vectorize (TF-IDF), apply PCA (where appropriate), and predict class labels; shows example predictions on sample tweets.

## Results
- The notebook contains evaluation and comparison code (classification reports, confusion matrices, and a comparison table). In this repository the notebook cells are present but may not have been executed — numeric results are produced when the notebook is run. The evaluation pipeline stores best estimators and selects the top model by weighted F1-score.

To reproduce results and generate the comparison table and model outputs, run the notebook (see commands below).

## How to run
1. Create a Python environment (recommended Python 3.8+).
2. Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Run the notebook interactively with Jupyter Lab / Notebook, or execute it end-to-end from the command line:

```bash
# From the repository root (note the space in filename)
jupyter nbconvert --to notebook --execute "Hate Speech Detection.ipynb" --inplace
```

This will execute the notebook in-place and write outputs (plots, metrics) into the `.ipynb` file.

## Example: programmatic prediction (after running notebook)
After training and selecting `best_model`, the notebook exposes a `predict_hate_speech` helper. Example usage:

```python
sample = "I hate these people, they should not exist"
pred = predict_hate_speech(sample, best_model, tfidf, pca)
print('Predicted class:', pred)
```

Note: For the Naïve Bayes model the notebook uses `X_tfidf` directly (no PCA) because PCA output can have negative values which NB cannot accept.

## Notes & Recommendations
- If you want to reproduce exact runs, ensure the notebook is executed in the same environment and that random seeds (`random_state=42`) are respected.
- For improved performance consider using pretrained word embeddings or transformer-based models (e.g., fine-tuned BERT) as future work.

## Contact
If you want me to run the notebook here and include the generated numeric results and plots in the README, tell me and I will execute it and update this file.
