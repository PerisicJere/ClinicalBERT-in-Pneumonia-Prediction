import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
from icd9cms.icd9 import search

df = pd.read_csv('cohorts/temporally_valid_pneumonia_cohort.tsv', sep='\t')
df.dropna(inplace=True)
df['ICD9_CODE_HISTORY'] = df['ICD9_CODE_HISTORY'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

groups = df['SUBJECT_ID'].values

icd9_dummies = df['ICD9_CODE_HISTORY'].str.join('|').str.get_dummies()
df = pd.concat([df.drop(columns=['ICD9_CODE_HISTORY']), icd9_dummies], axis=1)

categorical_columns = ['GENDER', 'ADMISSION_TYPE', 'ETHNICITY']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop(
    columns=['Pneumonia', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DOB', 'num_prior_codes', 'num_prior_admissions'],
    errors='ignore')
y = df['Pneumonia']

scaler = StandardScaler()
X = scaler.fit_transform(X)

selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

model = LogisticRegression(
    max_iter=1000,
    random_state=13,
    solver='saga',
    penalty='l2',
    C=np.float64(0.23357214690901212),
    class_weight='balanced'
)

gkf = GroupKFold(n_splits=5)

y_probs = np.zeros(len(y))
y_preds = np.zeros(len(y))

for train_idx, test_idx in gkf.split(X_selected, y, groups):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_probs[test_idx] = model.predict_proba(X_test)[:, 1]
    y_preds[test_idx] = model.predict(X_test)

auc_score = roc_auc_score(y, y_probs)
conf_matrix = confusion_matrix(y, y_preds)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)

n_bootstraps = 1000
rng = np.random.default_rng(seed=13)
bootstrapped_aucs = []

for _ in range(n_bootstraps):
    indices = rng.choice(range(len(y)), size=len(y), replace=True)
    y_resampled = y.iloc[indices]
    y_probs_resampled = y_probs[indices]
    bootstrapped_aucs.append(roc_auc_score(y_resampled, y_probs_resampled))

lower_ci = np.percentile(bootstrapped_aucs, 2.5)
upper_ci = np.percentile(bootstrapped_aucs, 97.5)

plt.figure(figsize=(10, 6))
auc_scores = []

for fold, (train_idx, test_idx) in enumerate(
        tqdm(gkf.split(X_selected, y, groups), desc="Cross-validation folds", ncols=100, total=5)):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_test_probs = model.predict_proba(X_test)[:, 1]

    fold_auc = roc_auc_score(y_test, y_test_probs)
    auc_scores.append(fold_auc)

    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC: {fold_auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Each Fold")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("roc_curve_logistic.png")
plt.show()

print(f"AUC: {auc_score:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"PPV (Precision): {ppv:.4f}")
print(f"NPV: {npv:.4f}")

model.fit(X_selected, y)

selected_features = pd.DataFrame(X, columns=df.drop(
    columns=['Pneumonia', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DOB', 'num_prior_codes', 'num_prior_admissions'],
    errors='ignore').columns).columns[selector.get_support()]

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
plt.savefig("feature_importance.png")
plt.show()

for i in sorted_idx:
    label = get_feature_label(selected_features[i])
    print(f"{label}: {perm_importance.importances_mean[i]:.4f}")