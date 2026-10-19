# 💰 Annual Income Prediction Using Machine Learning

## 📌 Description du projet

Ce projet a pour objectif de prédire le niveau de revenu annuel d'un individu à partir de différentes caractéristiques démographiques et professionnelles, en utilisant des techniques de Machine Learning.

Le système effectue une classification binaire afin de déterminer si le revenu d'une personne est **inférieur ou égal à 50 000 $ (`<=50K`)** ou **supérieur à 50 000 $ (`>50K`)**.

Le projet utilise le dataset **Adult Census Income** et compare plusieurs algorithmes de classification afin d'évaluer leurs performances et d'identifier le modèle le plus performant selon le F1-score.

## 🎯 Objectifs

- Analyser et préparer les données.
- Gérer les valeurs manquantes.
- Transformer les variables numériques et catégorielles.
- Entraîner plusieurs modèles de Machine Learning.
- Comparer les performances des modèles.
- Visualiser les résultats à l'aide de graphiques et de matrices de confusion.
- Identifier le modèle présentant le meilleur F1-score sur les données de test.

## 🧠 Algorithmes utilisés

Les modèles suivants sont évalués :

- **Extra Trees Classifier**
- **Random Forest Classifier**
- **Logistic Regression**
- **HistGradientBoosting Classifier**

## 🛠️ Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Git et GitHub

## 📂 Structure du projet

```text
projectDami/
│
├── index.py
├── adult.csv
├── README.md
├── .gitignore
│
└── resultats/
    ├── model_comparison_results.csv
    ├── confusion_matrix_best_model.png
    ├── roc_curve_best_model.png
    ├── model_comparison_visualization.png
    └── eda_analysis.png
```

*Le dossier `resultats/` et ses fichiers sont générés après l'exécution du programme.*

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/mounder10-Se/Prdiction-du-revenu-annuel.git
```

### 2. Accéder au dossier du projet

```bash
cd Prdiction-du-revenu-annuel
```

### 3. Installer les dépendances

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Vérifier les données

Placer le fichier `adult.csv` dans le même dossier que `index.py`.

### 5. Exécuter le programme

```bash
python index.py
```

## 📊 Évaluation des modèles

Les performances des algorithmes sont évaluées avec les métriques suivantes :

| Métrique | Description |
|---|---|
| Accuracy | Proportion totale de prédictions correctes |
| Precision | Proportion de prédictions positives correctes |
| Recall | Capacité à identifier les individus de la classe positive |
| F1-score | Moyenne harmonique de la précision et du rappel |
| ROC-AUC | Capacité du modèle à distinguer les deux classes |

Les données sont séparées en un ensemble d'entraînement et un ensemble de test, avec une proportion de 80 % pour l'entraînement et 20 % pour le test.

## 📈 Résultats et visualisations

Le programme génère automatiquement :

- `model_comparison_results.csv` : tableau comparatif des performances.
- `confusion_matrix_best_model.png` : matrice de confusion du meilleur modèle.
- `roc_curve_best_model.png` : courbe ROC du meilleur modèle, si disponible.
- `model_comparison_visualization.png` : comparaison graphique des métriques.
- `eda_analysis.png` : visualisation exploratoire des données.

Le meilleur modèle est sélectionné en fonction de son F1-score sur l'ensemble de test.

## 🚀 Améliorations possibles

- Optimiser les hyperparamètres des modèles.
- Tester des algorithmes supplémentaires.
- Améliorer le traitement des valeurs manquantes.
- Étudier l'importance des variables.
- Développer une interface web avec Flask ou Streamlit.
- Évaluer la robustesse des résultats avec une validation croisée.

## ⚠️ Limites

Les performances dépendent de la qualité et de la représentativité des données. Une bonne accuracy ne garantit pas à elle seule un modèle équitable ou fiable pour toutes les populations. Les résultats doivent être interprétés avec prudence, notamment pour les décisions liées aux revenus et aux personnes.

## 👨‍💻 Auteur

**Projet de Machine Learning — Prédiction du revenu annuel**

Développé avec Python et Scikit-learn.

## 📄 Licence

Aucune licence spécifique n'est définie pour ce projet pour le moment.
