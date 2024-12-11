import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
from icd9cms.icd9 import search
# Load and preprocess data
df = pd.read_csv('cohorts/cleaned_data_of_pneumonia_patients.csv')
df.dropna(inplace=True)
df['ICD9_CODE_HISTORY'] = df['ICD9_CODE_HISTORY'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
icd9_dummies = df['ICD9_CODE_HISTORY'].str.join('|').str.get_dummies()
df = pd.concat([df.drop(columns=['ICD9_CODE_HISTORY']), icd9_dummies], axis=1)

# One-hot encode categorical variables
categorical_columns = ['GENDER', 'ADMISSION_TYPE', 'ETHNICITY']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Prepare feature matrix (X) and target vector (y)
X = df.drop(columns=['Pneumonia', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DOB'])
y = df['Pneumonia']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Initialize logistic regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=13,
    solver='saga',
    penalty='l2',
    C=np.float64(0.23357214690901212)
)

# Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
y_probs = cross_val_predict(model, X_selected, y, cv=skf, method='predict_proba')[:, 1]
y_preds = cross_val_predict(model, X_selected, y, cv=skf)

# Calculate performance metrics
auc_score = roc_auc_score(y, y_probs)
conf_matrix = confusion_matrix(y, y_preds)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)

# Bootstrapping for AUC confidence interval
n_bootstraps = 1000
rng = np.random.default_rng(seed=13)
bootstrapped_aucs = []

for _ in range(n_bootstraps):
    indices = rng.choice(range(len(y)), size=len(y), replace=True)
    y_resampled = y.iloc[indices]
    y_probs_resampled = y_probs[indices]
    bootstrapped_aucs.append(roc_auc_score(y_resampled, y_probs_resampled))

# AUC confidence interval
lower_ci = np.percentile(bootstrapped_aucs, 2.5)
upper_ci = np.percentile(bootstrapped_aucs, 97.5)

# Plot ROC curves
plt.figure(figsize=(10, 6))
auc_scores = []
for fold, (train_index, test_index) in enumerate(
        tqdm(skf.split(X_selected, y), desc="Cross-validation folds", ncols=100)):
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_test_probs = model.predict_proba(X_test)[:, 1]

    fold_auc = roc_auc_score(y_test, y_test_probs)
    auc_scores.append(fold_auc)

    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC: {auc_score:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Each Fold")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("logistic_images/roc_curve_logistic.png")
plt.show()

# Print AUC and metrics
print(f"AUC: {auc_score:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"PPV (Precision): {ppv:.4f}")
print(f"NPV: {npv:.4f}")

# Fit the model to the full dataset
model.fit(X_selected, y)

# Get selected features
selected_features = pd.DataFrame(X, columns=df.drop(columns=['Pneumonia', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DOB']).columns).columns[selector.get_support()]

perm_importance = permutation_importance(
    model, X_selected, y, scoring='roc_auc', random_state=13, n_repeats=10
)

sorted_idx = perm_importance.importances_mean.argsort()[-20:]

def get_feature_label(feature_name):
    try:
        result = search(feature_name)
        return f"{result.code} {result.short_desc}" if result.short_desc else feature_name
    except:
        return feature_name

plt.figure(figsize=(12, 8))
feature_names = [get_feature_label(selected_features[i]) for i in sorted_idx]
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_names)
plt.xlabel("Mean Importance")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("logistic_images/feature_importance.png")
plt.show()

# Print feature importances
for i in sorted_idx:
    label = get_feature_label(selected_features[i])
    print(f"{label}: {perm_importance.importances_mean[i]:.4f}")