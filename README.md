# ClinicalBERT in Pneumonia Prediction
## Overview
This repository contains the code, data, and analysis for the study ClinicalBERT in Pneumonia Prediction: Achieving High Accuracy with Fine-Tuning. The project highlights the use of ClinicalBERT, Random Forest, and Logistic Regression models for predicting pneumonia using electronic health records (EHRs). ClinicalBERT demonstrated superior performance, underscoring the potential of transformer-based models in clinical settings.
### Introduction
Pneumonia is a significant global health challenge. Traditional diagnostic methods, while effective, can be slow and require specialized expertise. This project leverages ClinicalBERT, a language model pre-trained on clinical text, to predict pneumonia. The study compares ClinicalBERT with Random Forest and Logistic Regression models, demonstrating ClinicalBERT's superior accuracy across all evaluated metrics.
### Data Preprocessing
#### Dataset
The study utilized the publicly available MIMIC-III database. Key preprocessing steps included:
- Identifying pneumonia cases using ICD-9 codes.
- Excluding incorrect or missing ICD-9 codes.
- Including patient demographics, such as age, ethnicity, and admission type.
#### Key Steps
- ICD-9 Code Transformation
  - For Random Forest and Logistic Regression: One-hot encoding of ICD-9 codes.
  - For ClinicalBERT: Tokenization using AutoTokenizer.
- Data Splitting
  - Applied five-fold cross-validation to ensure robust evaluation.
### Models and Training
#### Random Forest
- Hyperparameters: Grid search optimization for parameters such as n_estimators and max_depth.
- Framework: scikit-learn.
#### Logistic Regression
- Hyperparameters: Regularization with C=0.23357 and solver='saga'.
- Framework: scikit-learn.
#### ClinicalBERT
- Pre-trained Model: ClinicalBERT fine-tuned for pneumonia prediction.
- Key Hyperparameters:
  - Learning rate: 2e-5
  - Epochs: 7
  - Batch size: 16
  - Max sequence length: 128 tokens
### Evaluation Metrics
- Area Under the Receiver Operating Characteristic Curve (AUC)
- Sensitivity
- Specificity
- Positive Predictive Value (PPV)
- Negative Predictive Value (NPV)
### Results
| Model               | AUC    | Sensitivity | Specificity | PPV    | NPV    |
|---------------------|--------|-------------|-------------|--------|--------|
| Random Forest       | 0.9840 | 0.9481      | 0.9230      | 0.8945 | 0.9628 |
| Logistic Regression | 0.9418 | 0.9364      | 0.8338      | 0.7949 | 0.9502 |
| ClinicalBERT        | 0.9974 | 0.9792      | 0.9816      | 0.9744 | 0.9852 |

#### Key Insights
- ClinicalBERT achieved near-perfect AUC scores across all folds.
- Random Forest demonstrated strong performance with interpretable feature importance.
### Paper
[ClinicalBERT in Pneumonia Prediction: Achieving High Accuracy with Fine Tuning.pdf](ClinicalBERT_in_Pneumonia_Prediction__Achieving_High_Accuracy_with_Fine_Tuning.pdf)
