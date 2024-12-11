import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve, auc
)

# Custom Dataset class for loading pneumonia data for sequence classification.
class PneumoniaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

# Converts a DataFrame to a dataset that can be used for training/evaluation
def create_dataset(df, tokenizer):
    texts = [str(x) for x in df['ICD9_CODE_HISTORY'].tolist()]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    labels = torch.tensor(df['Pneumonia'].values, dtype=torch.long).cpu()
    return PneumoniaDataset(encodings, labels)

# Computes detailed performance metrics including Sensitivity, Specificity,
# PPV (Positive Predictive Value), NPV (Negative Predictive Value), AUC, and
# their 95% confidence intervals using the Wilson score interval for each metric.
def detailed_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    auc = roc_auc_score(y_true, y_prob)

    def wilson_ci(p, n, z=1.96):
        denominator = 1 + z ** 2 / n
        center_adjusted_probability = p + z * z / (2 * n)
        adjusted_standard_deviation = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        lower_bound = (center_adjusted_probability - adjusted_standard_deviation) / denominator
        upper_bound = (center_adjusted_probability + adjusted_standard_deviation) / denominator
        return (lower_bound, upper_bound)

    sens_ci = wilson_ci(sensitivity, len(y_true))
    spec_ci = wilson_ci(specificity, len(y_true))
    ppv_ci = wilson_ci(ppv, len(y_true))
    npv_ci = wilson_ci(npv, len(y_true))
    auc_ci = (auc - 1.96 * np.sqrt((auc * (1 - auc)) / len(y_true)),
              auc + 1.96 * np.sqrt((auc * (1 - auc)) / len(y_true)))

    return {
        'AUC': auc,
        'AUC_95%_CI': auc_ci,
        'Sensitivity': sensitivity,
        'Sensitivity_95%_CI': sens_ci,
        'Specificity': specificity,
        'Specificity_95%_CI': spec_ci,
        'PPV': ppv,
        'PPV_95%_CI': ppv_ci,
        'NPV': npv,
        'NPV_95%_CI': npv_ci,
    }

# Computes evaluation metrics for a model's predictions.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()

    metrics = detailed_metrics(labels, preds, probs)
    print("\nDetailed Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics
# The main function to initialize the model, tokenizer, train on the dataset,
# evaluate performance on each fold, and save the trained model and metrics.
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "medicalai/ClinicalBERT",
        num_labels=2
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    df = pd.read_csv("cohorts/cleaned_data_of_pneumonia_patients.csv")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

    auc_values = []
    sens_values = []
    spec_values = []
    ppv_values = []
    npv_values = []

    all_fpr = []
    all_tpr = []
    all_roc_auc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Pneumonia'])):
        print(f"\nTraining fold {fold + 1}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = create_dataset(train_df, tokenizer)
        val_dataset = create_dataset(val_df, tokenizer)

        training_args = TrainingArguments(
            output_dir=f'./output/fold_{fold + 1}',
            num_train_epochs=7,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy='epoch',
            eval_steps=500,
            save_steps=500,
            logging_dir='./logs',
            learning_rate=2e-5,
            save_total_limit=1,
            metric_for_best_model='auc',
            logging_steps=100,
            warmup_steps=500,
            gradient_accumulation_steps=2,
            fp16=True,
            max_grad_norm=1.0
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        try:
            trainer.train()

            val_results = trainer.evaluate(eval_dataset=val_dataset)
            print(f"\n{val_results}")
            all_fpr, all_tpr, all_roc_auc = plot_auc(trainer, val_dataset, fold, all_fpr, all_tpr, all_roc_auc)

            eval_preds = trainer.predict(val_dataset)
            labels = eval_preds.label_ids
            preds = eval_preds.predictions.argmax(-1)
            probs = torch.softmax(torch.tensor(eval_preds.predictions), dim=-1)[:,
                    1].numpy()

            metrics = detailed_metrics(labels, preds, probs)
            auc_values.append(metrics['AUC'])
            sens_values.append(metrics['Sensitivity'])
            spec_values.append(metrics['Specificity'])
            ppv_values.append(metrics['PPV'])
            npv_values.append(metrics['NPV'])

            model.save_pretrained(f'./output/final_model_fold_{fold + 1}')
            tokenizer.save_pretrained(f'./output/final_model_fold_{fold + 1}')

        except Exception as e:
            print(f"Error during training (Fold {fold + 1}): {str(e)}")
            raise

    fold_means = compute_fold_means(auc_values, sens_values, spec_values, ppv_values, npv_values)
    print("\nMean Metrics across all folds:")
    for key, value in fold_means.items():
        print(f"{key}: {value:.4f}")

    plot_all_folds_auc(all_fpr, all_tpr, all_roc_auc)

# Plots the ROC curve for the given fold and adds the results to the cumulative AUC values.
def plot_auc(trainer, val_dataset, fold, all_fpr, all_tpr, all_roc_auc):
    eval_preds = trainer.predict(val_dataset)
    labels = eval_preds.label_ids
    probs = torch.softmax(torch.tensor(eval_preds.predictions), dim=-1)[:, 1].numpy()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

    return all_fpr, all_tpr, all_roc_auc

# Plots the aggregated ROC curve for all folds.
def plot_all_folds_auc(all_fpr, all_tpr, all_roc_auc):
    plt.figure(figsize=(8, 6))

    for i in range(len(all_fpr)):
        plt.plot(all_fpr[i], all_tpr[i], lw=2, label=f'Fold {i + 1} (AUC: {all_roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Folds')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
# Computes the mean values of all evaluation metrics across the 5 folds.
def compute_fold_means(auc_values, sens_values, spec_values, ppv_values, npv_values):
    mean_auc = np.mean(auc_values)
    mean_sensitivity = np.mean(sens_values)
    mean_specificity = np.mean(spec_values)
    mean_ppv = np.mean(ppv_values)
    mean_npv = np.mean(npv_values)

    return {
        'Mean AUC': mean_auc,
        'Mean Sensitivity': mean_sensitivity,
        'Mean Specificity': mean_specificity,
        'Mean PPV': mean_ppv,
        'Mean NPV': mean_npv
    }

if __name__ == "__main__":
    main()