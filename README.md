# Détection de Fraude par Carte Bancaire

Implémentation complète d'un système de détection de fraude bancaire utilisant des techniques avancées de machine learning et de gestion du déséquilibre des classes.

## Contexte

La fraude par carte bancaire représente un enjeu majeur pour les institutions financières. Ce projet vise à développer un système de détection automatique capable d'identifier les transactions frauduleuses en temps réel.

## Points de Différenciation

Ce projet se distingue par plusieurs aspects méthodologiques et business :

**1. Comparaison Systématique des Approches**
- Évaluation de 4 techniques de rééchantillonnage (SMOTE, random under-sampling, SMOTE+Tomek Links, class weighting)
- Test de 4 algorithmes complémentaires (régression logistique, Random Forest, XGBoost, Isolation Forest)
- Analyse des combinaisons optimales selon les contraintes métier

**2. Perspective Business Intégrée**
- Analyse coût-bénéfice avec hypothèses réalistes (coût fraude non détectée vs faux positif)
- Recommandation basée sur l'impact économique, pas uniquement les métriques ML
- Calcul du ROI pour chaque approche

**3. Rigueur Méthodologique**
- Feature engineering temporel (création de variables horaires, périodes, week-end)
- Normalisation adaptée avec RobustScaler (résistant aux outliers)
- Validation stratifiée maintenant le ratio fraude/légitime
- 7 métriques d'évaluation adaptées au déséquilibre

**4. Code Production-Ready**
- Structure modulaire en 3 notebooks séparés (EDA, Preprocessing, Modeling)
- Pipeline reproductible avec sauvegarde des transformations
- Modèles et scaler persistés pour déploiement
- Documentation complète et commentaires détaillés

## Objectifs

- Analyser les patterns de transactions frauduleuses
- Comparer plusieurs algorithmes de machine learning
- Proposer un modèle optimisé selon des critères business
- Évaluer l'impact économique des différentes approches

## Dataset

Source : [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Caractéristiques :
- 284 807 transactions
- 492 fraudes (0.17%)
- 30 features (28 issues de PCA + Time + Amount)
- Données anonymisées pour raisons de confidentialité

## Méthodologie

### 1. Analyse Exploratoire (EDA)

J'ai commencé par une analyse approfondie du dataset pour comprendre :
- La distribution des classes (déséquilibre important : ratio 1:578)
- Les patterns temporels des fraudes
- La distribution des montants selon le type de transaction
- Les corrélations entre variables

Constats principaux :
- Déséquilibre extrême nécessitant des techniques de rééchantillonnage
- Certaines composantes PCA montrent une séparation claire entre fraudes et transactions légitimes
- Les montants des transactions frauduleuses diffèrent significativement

### 2. Preprocessing

Étapes réalisées :
- Split stratifié train/test (80/20)
- Feature engineering : création de variables temporelles (heure, jour, période)
- Normalisation avec RobustScaler (résistant aux outliers)
- Gestion du déséquilibre avec 4 techniques :
  - SMOTE
  - Random Under-sampling
  - SMOTE + Tomek Links
  - Class weighting

### 3. Modélisation

J'ai testé 4 algorithmes différents :

**Régression Logistique** (baseline)
- Approche simple et interprétable
- Utilisée avec class_weight='balanced'
- Meilleur recall : 90.82%

**Random Forest**
- Entraîné sur données SMOTE
- 100 arbres, max_depth=20
- Meilleur F1-Score : 0.7864

**XGBoost**
- Utilisation de scale_pos_weight
- Paramètres optimisés pour le déséquilibre
- Meilleur ROC-AUC : 0.9824

**Isolation Forest**
- Approche par détection d'anomalies
- Contamination ajustée au ratio de fraudes
- Baseline pour comparaison

## Résultats

### Performances des Modèles

| Modèle | Precision | Recall | F1-Score | ROC-AUC | Coût Business |
|--------|-----------|--------|----------|---------|---------------|
| Logistic Regression | 0.6136 | 0.9082 | 0.7319 | 0.9742 | 1,860€ |
| Random Forest | 0.7500 | 0.8265 | 0.7864 | 0.9793 | 2,026€ |
| **XGBoost** | **0.7222** | **0.8367** | **0.7752** | **0.9824** | **1,566€** |
| Isolation Forest | 0.0408 | 0.8878 | 0.0781 | 0.9366 | 27,322€ |

### Analyse Business

Hypothèses de coût :
- Fraude non détectée (FN) : 100€
- Faux positif à vérifier (FP) : 2€

**Recommandation : XGBoost**
- Meilleur compromis performance/coût
- ROC-AUC de 98.24% (excellente discrimination)
- Coût total minimal : 1,566€
- 83.67% de recall (capture la majorité des fraudes)

## Structure du Projet

```
fraud-detection-banking/
│
├── data/
│   ├── raw/                      # Données brutes
│   └── processed/                # Données prétraitées
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Analyse exploratoire
│   ├── 02_Preprocessing.ipynb    # Préparation des données
│   └── 03_Modeling.ipynb         # Modélisation et évaluation
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── isolation_forest_model.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── figures/                  # Graphiques et visualisations
│   ├── models_comparison.csv     # Tableau comparatif
│   └── eda_summary.json          # Statistiques clés
│
├── README.md
└── requirements.txt
```

## Technologies Utilisées

- **Python 3.9+**
- **Pandas, NumPy** : manipulation de données
- **Scikit-learn** : modélisation et preprocessing
- **XGBoost** : gradient boosting
- **Imbalanced-learn** : gestion du déséquilibre
- **Matplotlib, Seaborn** : visualisations
- **Joblib** : sauvegarde des modèles

## Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/fraud-detection-banking.git
cd fraud-detection-banking

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset depuis Kaggle
# Placer creditcard.csv dans data/raw/
```

## Utilisation

```python
import joblib
import pandas as pd

# Charger le modèle et le scaler
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Préparer les données
# (normalisation + feature engineering nécessaires)

# Prédiction
prediction = model.predict(X_new)
proba = model.predict_proba(X_new)[:, 1]
```

## Visualisations

Le projet inclut 17 graphiques détaillés :
- Distribution des classes et déséquilibre
- Analyse temporelle des fraudes
- Corrélations avec la variable cible
- Matrices de confusion par modèle
- Courbes ROC et Precision-Recall
- Importance des features
- Analyse comparative des performances
- Coûts business par modèle

## Perspectives d'Amélioration

1. **Hyperparameter Tuning**
   - GridSearch sur XGBoost
   - Optimisation bayésienne

2. **Ensemble Methods**
   - Voting Classifier
   - Stacking de modèles

3. **Interprétabilité**
   - SHAP values
   - LIME pour expliquer les prédictions

4. **Mise en Production**
   - API REST (FastAPI)
   - Dashboard de monitoring
   - Pipeline automatisé

## Auteur

** Daniel Kpakpa **  
Étudiant en Master 2 Ingénierie des Données et Évaluation Économétrique  
Contact : danielkpakpapro@gmail.com 
LinkedIn : [Daniel Kpakpa](https://www.linkedin.com/in/daniel-kpakpa-0101b6188/)  
GitHub : [danielkpakpapro-cmd](https://github.com/danielkpakpapro-cmd)

## Remerciements

- Dataset fourni par l'Université Libre de Bruxelles (Machine Learning Group)
- Kaggle pour l'hébergement des données

---

*Projet personnel - 2025*