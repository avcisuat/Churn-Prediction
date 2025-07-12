import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv(r"C:\Users\Pc\Downloads\Telco-Customer-Churn.csv")

# 2. Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 3. Data visualization
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn Distribution by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges vs. Churn')
plt.tight_layout()
plt.show()


# Tenure vs. Churn Rate
tenure_churn = df.groupby('tenure')['Churn'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=tenure_churn, x='tenure', y='Churn')
plt.title("Customer Tenure vs. Churn Rate")
plt.xlabel("Tenure (months)")
plt.ylabel("Churn Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Feature and target split
X = df.drop('Churn', axis=1)
y = df['Churn']
X = pd.get_dummies(X, drop_first=True)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Scaling + SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 7. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# 8. GridSearchCV + Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5],
    'clf__class_weight': ['balanced', None]
}

grid_search = GridSearchCV(pipeline, param_grid, scoring='recall', cv=5)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
models["Random Forest"] = best_rf

#denemee
# 9. Heatmap ile GridSearch görselleştirme (geliştirilmiş versiyon)
results = pd.DataFrame(grid_search.cv_results_)
pivot = results.pivot_table(
    index='param_clf__max_depth',
    columns='param_clf__n_estimators',
    values='mean_test_score'
)

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
plt.title("GridSearchCV Heatmap – Mean Recall Scores")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.tight_layout()
plt.show()


# 9. GridSearchCV visualization
def plot_gridsearch_scatter(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results,
        x='param_clf__n_estimators',
        y='param_clf__max_depth',
        size='mean_test_score',
        hue='mean_test_score',
        palette='viridis',
        sizes=(40, 200)
    )
    plt.title("GridSearch Results - n_estimators vs. max_depth")
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.legend(title="Mean Test Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_gridsearch_scatter(grid_search)

# 10. Feature Importance
def plot_feature_importances(model, feature_names, top_n=10):
    importances = model.named_steps['clf'].feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    top_features = [(feature_names[i], importances[i]) for i in indices]
    feat_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_df)
    plt.title("Random Forest - Top Important Features")
    plt.tight_layout()
    plt.show()

plot_feature_importances(best_rf, X.columns)

# 11. Model training and evaluation
results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Report": classification_report(y_test, y_pred, output_dict=True)
    }

# 12. Performance summary
summary_df = pd.DataFrame({
    model_name: {
        "Accuracy": round(metrics["Accuracy"], 2),
        "Precision": round(metrics["Report"]["1"]["precision"], 2),
        "Recall": round(metrics["Report"]["1"]["recall"], 2),
        "F1-Score": round(metrics["Report"]["1"]["f1-score"], 2)
    }
    for model_name, metrics in results.items()
}).T
print("\nModel Performance Summary:\n", summary_df)

# 13. Logistic Regression threshold adjustment
lr_model = models["Logistic Regression"]
y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.3
y_pred_thresh = (y_proba >= threshold).astype(int)
print(f"\nLogistic Regression with threshold={threshold} classification report:")
print(classification_report(y_test, y_pred_thresh))

# 14. ROC + Confusion Matrix
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

