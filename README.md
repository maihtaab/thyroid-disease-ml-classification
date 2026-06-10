# Thyroid Disease ML Classification

Machine learning project for detecting hyperthyroid disease using clinical patient data from the UCI Thyroid Disease Dataset.

## Overview

This project evaluates the effectiveness of supervised machine learning models for thyroid disease detection. The dataset was preprocessed through missing value handling, categorical feature encoding, and train-test splitting before model training.

Three classification models were implemented and compared:

* Logistic Regression
* Decision Tree
* Random Forest

The objective was to determine which model provides the most reliable performance for identifying hyperthyroid disease cases.

## Dataset

Source: UCI Machine Learning Repository – Thyroid Disease Dataset

The dataset contains patient demographic information, laboratory test results, and diagnostic indicators related to thyroid disorders.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Jupyter Notebook

## Results

| Model               | Accuracy   | Precision | Recall    | F1 Score  |
| ------------------- | ---------- | --------- | --------- | --------- |
| Logistic Regression | 98.39%     | 87.5%     | 46.7%     | 60.9%     |
| Decision Tree       | 98.39%     | 71.4%     | 66.7%     | 69.0%     |
| Random Forest       | **98.75%** | **83.3%** | **66.7%** | **74.1%** |

The Random Forest classifier achieved the strongest overall performance, producing the highest accuracy and F1 score while maintaining strong precision and recall.

## Visualizations

### Model Performance Comparison

![Model Performance](reports/figures/model_performance_comparison.png)

### Random Forest Confusion Matrix

![Random Forest Confusion Matrix](reports/figures/random_forest_confusion_matrix.png)

## Full Project Report

A detailed report describing the methodology, preprocessing pipeline, model evaluation, and conclusions can be found here:

**reports/Machine-Learning-Approaches-for-Thyroid-Disease-Detection.pdf**

## Repository Structure

```text
src/                    # Source code
reports/                # Project report and visualizations
notebooks/              # Jupyter notebooks
requirements.txt        # Python dependencies
```

## Author

Maihtaab Sidhu

Wilfrid Laurier University
Bachelor of Science in Computer Science

