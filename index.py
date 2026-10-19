
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("adult.csv")

df = dataset.copy()

df = df.replace(' ?', np.nan)

df['income'] = df['income'].apply(lambda x: 0 if x.strip() == '<=50K' else 1)

print("Distribution des classes:")
print(df['income'].value_counts())
print("\nNombre de valeurs manquantes par colonne:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlations = df[numeric_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0])
axes[0].set_title('Heatmap des corrélations')
df['income'].value_counts().plot(kind='bar', color=['skyblue', 'orange'], ax=axes[1])
axes[1].set_title('Distribution des classes (income)')
axes[1].set_ylabel('Nombre de personnes')
plt.tight_layout()
plt.savefig('eda_analysis.png')
plt.close()

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('income') if 'income' in categorical_cols else None

df_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].astype(str)
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

simple_imputer = SimpleImputer(strategy='most_frequent')
X_train_simple = simple_imputer.fit_transform(X_train)
X_test_simple = simple_imputer.transform(X_test)

knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = knn_imputer.fit_transform(X_train)
X_test_knn = knn_imputer.transform(X_test)

scaler = StandardScaler()
X_train_simple_scaled = scaler.fit_transform(X_train_simple)
X_test_simple_scaled = scaler.transform(X_test_simple)

scaler_knn = StandardScaler()
X_train_knn_scaled = scaler_knn.fit_transform(X_train_knn)
X_test_knn_scaled = scaler_knn.transform(X_test_knn)

models = {
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)
}

results = []

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, imputation_method):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'Model': model_name,
        'Imputation': imputation_method,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

print("\nÉvaluation des modèles avant optimisation:")
for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    result_simple = evaluate_model(model, X_train_simple_scaled, X_test_simple_scaled, y_train, y_test, model_name, 'Simple')
    results.append(result_simple)
    print(f"  Simple Imputation - Accuracy: {result_simple['Accuracy']:.4f}, F1: {result_simple['F1-Score']:.4f}")
    
    result_knn = evaluate_model(model, X_train_knn_scaled, X_test_knn_scaled, y_train, y_test, model_name, 'KNN')
    results.append(result_knn)
    print(f"  KNN Imputation - Accuracy: {result_knn['Accuracy']:.4f}, F1: {result_knn['F1-Score']:.4f}")

param_grids = {
    'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'NaiveBayes': {},
    'DecisionTree': {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20], 'min_samples_split': [2, 5]}
}

results_optimized = []

print("\n\nOptimisation des hyperparamètres:")
for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    if param_grids[model_name]:
        grid_simple = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1', n_jobs=-1)
        grid_simple.fit(X_train_simple_scaled, y_train)
        best_model_simple = grid_simple.best_estimator_
        print(f"  Meilleurs paramètres (Simple): {grid_simple.best_params_}")
    else:
        best_model_simple = model
    
    result_simple_opt = evaluate_model(best_model_simple, X_train_simple_scaled, X_test_simple_scaled, y_train, y_test, model_name, 'Simple_Optimized')
    results_optimized.append(result_simple_opt)
    print(f"  Simple Imputation Optimized - Accuracy: {result_simple_opt['Accuracy']:.4f}, F1: {result_simple_opt['F1-Score']:.4f}")
    
    if param_grids[model_name]:
        grid_knn = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1', n_jobs=-1)
        grid_knn.fit(X_train_knn_scaled, y_train)
        best_model_knn = grid_knn.best_estimator_
        print(f"  Meilleurs paramètres (KNN): {grid_knn.best_params_}")
    else:
        best_model_knn = model
    
    result_knn_opt = evaluate_model(best_model_knn, X_train_knn_scaled, X_test_knn_scaled, y_train, y_test, model_name, 'KNN_Optimized')
    results_optimized.append(result_knn_opt)
    print(f"  KNN Imputation Optimized - Accuracy: {result_knn_opt['Accuracy']:.4f}, F1: {result_knn_opt['F1-Score']:.4f}")

results_df = pd.DataFrame(results + results_optimized)
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n\nRésultats sauvegardés dans 'model_comparison_results.csv'")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    results_df_pivot = results_df.pivot(index='Model', columns='Imputation', values=metric)
    results_df_pivot.plot(kind='bar', ax=ax)
    ax.set_title(f'Comparaison - {metric}')
    ax.set_ylabel(metric)
    ax.set_xlabel('Model')
    ax.legend(title='Imputation Method')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_visualization.png')
plt.close()

print("\nGraphiques sauvegardés dans 'model_comparison_visualization.png'")

best_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
best_model.fit(X_train_knn_scaled, y_train)
y_pred_best = best_model.predict(X_test_knn_scaled)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion - Meilleur modèle')
plt.ylabel('Vrai')
plt.xlabel('Prédit')
plt.savefig('confusion_matrix_best_model.png')
plt.close()

y_pred_proba_best = best_model.predict_proba(X_test_knn_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Meilleur modèle')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve_best_model.png')
plt.close()

print("\n\nAnalyse terminée avec succès!")
print(f"Meilleure performance: {results_df.loc[results_df['F1-Score'].idxmax()]}")