
# ============================================================
# PROJET : PRÉDICTION DU REVENU ANNUEL
# Objectif : prédire si le revenu est <=50K ou >50K
# Fichier de données : adult.csv
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_FILE = "adult.csv"

OUTPUT_DIR = "resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES
# ------------------------------------------------------------

print("=" * 65)
print("PRÉDICTION DU REVENU ANNUEL")
print("=" * 65)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Fichier '{DATA_FILE}' introuvable. "
        "Place adult.csv dans le même dossier que index.py."
    )

df = pd.read_csv(DATA_FILE, skipinitialspace=True)

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# Remplacer les valeurs manquantes courantes du dataset Adult
df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
df = df.replace(r"^\s*$", np.nan, regex=True)

print("\nDimensions du dataset :", df.shape)
print("\nColonnes détectées :")
print(df.columns.tolist())

# ------------------------------------------------------------
# 3. IDENTIFICATION DE LA COLONNE CIBLE
# ------------------------------------------------------------

# Le dataset Adult existe sous plusieurs variantes.
# On accepte quelques noms fréquents pour la cible.

target_candidates = [
    "income",
    "class",
    "salary",
    "revenu"
]

target_col = next(
    (col for col in target_candidates if col in df.columns.str.lower()),
    None
)

if target_col is None:
    raise ValueError(
        "Impossible de trouver la colonne cible. "
        "Vérifie si la colonne du revenu s'appelle 'income' ou 'class'. "
        f"Colonnes présentes : {df.columns.tolist()}"
    )

# Conserver le nom réel de la colonne, même si sa casse diffère
target_col = next(
    col for col in df.columns
    if col.lower() == target_col
)

print("\nColonne cible :", target_col)

# Nettoyage des libellés de la cible
df[target_col] = (
    df[target_col]
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
)

# Conversion en classification binaire :
# 0 = revenu <=50K
# 1 = revenu >50K

target_mapping = {
    "<=50K": 0,
    ">50K": 1,
    "<=50K.": 0,
    ">50K.": 1
}

df[target_col] = df[target_col].map(target_mapping)

# Supprimer les lignes dont la cible est inconnue
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

if df[target_col].nunique() != 2:
    raise ValueError(
        "La cible doit contenir exactement deux classes : "
        "<=50K et >50K. Vérifie les valeurs de la colonne cible."
    )

print("\nRépartition des classes :")
print(df[target_col].value_counts().rename(index={
    0: "<=50K",
    1: ">50K"
}))

# ------------------------------------------------------------
# 4. PRÉPARATION DES VARIABLES
# ------------------------------------------------------------

X = df.drop(columns=[target_col])
y = df[target_col]

# Enlever les colonnes totalement vides
empty_columns = X.columns[X.isna().all()].tolist()

if empty_columns:
    print("\nColonnes entièrement vides supprimées :", empty_columns)
    X = X.drop(columns=empty_columns)

# Détecter les colonnes numériques et catégorielles
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNombre de variables numériques :", len(numeric_features))
print("Nombre de variables catégorielles :", len(categorical_features))

# ------------------------------------------------------------
# 5. SÉPARATION EN ENSEMBLES D'ENTRAÎNEMENT ET DE TEST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTaille de l'entraînement :", X_train.shape)
print("Taille du test :", X_test.shape)

# ------------------------------------------------------------
# 6. PRÉTRAITEMENT DES DONNÉES
# ------------------------------------------------------------

# Pour les modèles compatibles avec une représentation sparse.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_sparse = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# HistGradientBoosting nécessite une matrice dense.
# Ce préprocesseur lui est réservé pour éviter de convertir
# inutilement les données de tous les autres modèles en dense.

categorical_transformer_dense = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor_dense = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer_dense, categorical_features)
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# 7. DÉFINITION DES MODÈLES
# ------------------------------------------------------------

models = {
    "Extra Trees": Pipeline(steps=[
        ("preprocessor", preprocessor_sparse),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor_sparse),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2
        ))
    ]),

    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor_sparse),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Hist Gradient Boosting": Pipeline(steps=[
        ("preprocessor", preprocessor_dense),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE
        ))
    ])
}

# ------------------------------------------------------------
# 8. ENTRAÎNEMENT ET ÉVALUATION
# ------------------------------------------------------------

results = []
trained_models = {}
predictions = {}
probabilities = {}

print("\n" + "=" * 65)
print("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
print("=" * 65)

for model_name, model in models.items():
    print(f"\nEntraînement : {model_name}")

    try:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Certaines versions de modèles peuvent ne pas proposer
        # predict_proba ; dans ce cas, ROC-AUC reste indisponible.
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        roc_auc = (
            roc_auc_score(y_test, y_proba)
            if y_proba is not None
            else np.nan
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        trained_models[model_name] = model
        predictions[model_name] = y_pred

        if y_proba is not None:
            probabilities[model_name] = y_proba

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

    except Exception as error:
        print(f"Erreur pour {model_name} : {error}")

# ------------------------------------------------------------
# 9. COMPARAISON DES RÉSULTATS
# ------------------------------------------------------------

if not results:
    raise RuntimeError(
        "Aucun modèle n'a pu être entraîné. "
        "Vérifie le dataset et les erreurs affichées."
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
).reset_index(drop=True)

results_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 65)
print("COMPARAISON DES MODÈLES")
print("=" * 65)
print(results_df.to_string(index=False, float_format="%.4f"))

# Choisir le meilleur modèle selon le F1-score
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print("\nMeilleur modèle selon le F1-score :", best_model_name)
print(
    "Accuracy du meilleur modèle : "
    f"{accuracy_score(y_test, best_predictions):.4f}"
)

print("\nRapport de classification :")
print(classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["<=50K", ">50K"],
    zero_division=0
))

# ------------------------------------------------------------
# 10. MATRICE DE CONFUSION DU MEILLEUR MODÈLE
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["<=50K", ">50K"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(f"Matrice de confusion - {best_model_name}")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_best_model.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 11. COURBE ROC DU MEILLEUR MODÈLE
# ------------------------------------------------------------

if best_model_name in probabilities:
    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities[best_model_name],
        ax=ax,
        name=best_model_name
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Classification aléatoire"
    )

    ax.set_title(f"Courbe ROC - {best_model_name}")
    ax.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve_best_model.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ------------------------------------------------------------
# 12. VISUALISATION DE COMPARAISON DES MODÈLES
# ------------------------------------------------------------

metrics_to_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score"
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric",
    ax=ax
)

ax.set_title("Comparaison des performances des modèles")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_xlabel("Modèle")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Métrique")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison_visualization.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 13. ANALYSE EXPLORATOIRE SIMPLE
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Distribution des classes
class_counts = y.value_counts().sort_index()

axes[0].bar(
    ["<=50K", ">50K"],
    [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
)
axes[0].set_title("Répartition des revenus")
axes[0].set_ylabel("Nombre d'individus")

# Valeurs manquantes par colonne
missing_percent = (
    df.isna().mean() * 100
).sort_values(ascending=False)

missing_percent = missing_percent[missing_percent > 0].head(15)

if not missing_percent.empty:
    axes[1].barh(
        missing_percent.index.astype(str),
        missing_percent.values
    )
    axes[1].set_title("Pourcentage de valeurs manquantes")
    axes[1].set_xlabel("Pourcentage (%)")
    axes[1].invert_yaxis()
else:
    axes[1].text(
        0.5, 0.5,
        "Aucune valeur manquante détectée",
        ha="center",
        va="center"
    )
    axes[1].set_axis_off()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "eda_analysis.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 14. RÉSUMÉ FINAL
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EXÉCUTION TERMINÉE")
print("=" * 65)

print("Meilleur modèle :", best_model_name)
print(
    "Accuracy : "
    f"{accuracy_score(y_test, best_predictions):.4f}"
)
print(
    "F1-score : "
    f"{f1_score(y_test, best_predictions, zero_division=0):.4f}"
)

print(f"\nLes résultats sont enregistrés dans : {OUTPUT_DIR}")
print("Fichiers générés :")
print("- model_comparison_results.csv")
print("- confusion_matrix_best_model.png")
print("- roc_curve_best_model.png (si disponible)")
print("- model_comparison_visualization.png")
print("- eda_analysis.png")
