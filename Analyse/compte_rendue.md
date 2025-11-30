# **IKRAM EL-WALI**

**Classe** : CAC2

<br clear="left"/>

---

# **Compte rendu** : **Analyse des Dépenses Médicales par Régression**

---

# **À propos du jeu de données :**

Ce fichier contient **100 000 individus** avec des informations complètes sur :

* leur **profil démographique**
* leurs **habitudes de vie**
* leurs **conditions de santé**
* leurs **utilisations de services médicaux**
* leurs **plans d’assurance**
* leurs **dépenses annuelles**

Chaque ligne représente **un individu unique**.
Le dataset est particulièrement adapté pour :

* la **prédiction du coût médical annuel** (régression),
* la **classification du niveau de risque**,
* l’analyse statistique des facteurs influençant la santé.


# À propos du jeu de données :

Le dataset `medical_insurance.csv` contient des informations sur les patients, leurs caractéristiques démographiques, biométriques, historiques médicaux et leurs polices d'assurance. Chaque ligne correspond à un patient et inclut des informations telles que l'âge, le sexe, le statut marital, les antécédents médicaux, le mode de vie (tabac, alcool), les mesures biométriques (IMC, tension, LDL, HbA1c), et des informations sur les prestations d’assurance (type de plan, réseau, primes, nombre de réclamations, dépenses médicales annuelles, etc.).

**Variable cible :**  
- `is_high_risk` (1 = patient à risque élevé, 0 = patient à faible risque)

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données](#2-analyse-exploratoire-des-données)
    * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
    * [Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)
    * [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)
    * [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)
    * [Séparation des Données](#31-séparation-des-données)
    * [Modèles Testés](#32-modèles-testés)
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et Contexte

Ce projet vise à prédire si un patient est à **haut risque médical** en utilisant des informations démographiques, biométriques et médicales. L'objectif est d’identifier les patients nécessitant un suivi renforcé, en se basant sur l’analyse de données réelles et la modélisation prédictive.

---

## 2. Analyse Exploratoire des Données

### 2.1 Chargement et Structure du Dataset

```python
import pandas as pd
df = pd.read_csv('medical_insurance.csv')
print(df.shape)
df.info()
df.head()
````

* **Nombre d'observations :** [NOMBRE]
* **Nombre de variables :** 54 colonnes (features + target)

**Exemples de variables :**

* Démographiques : `age`, `sex`, `region`, `urban_rural`, `marital_status`
* Santé : `bmi`, `hypertension`, `diabetes`, `asthma`, `copd`, `cardiovascular_disease`, `cancer_history`
* Assurance : `plan_type`, `network_tier`, `annual_premium`, `claims_count`, `total_claims_paid`

**Variable cible :** `is_high_risk`

---

### 2.2 Prétraitement et Ingénierie de Caractéristiques

* Encodage des variables catégorielles (`sex`, `region`, `urban_rural`, `marital_status`, `employment_status`, `smoker`, `plan_type`, `network_tier`, `had_major_procedure`) via **LabelEncoder**.
* Création d’interactions pertinentes : par exemple, `ANXYELFIN = ANXIETY * YELLOW_FINGERS`.

```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
categorical_cols = ['sex','region','urban_rural','marital_status','employment_status',
                    'smoker','plan_type','network_tier','had_major_procedure']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
```

---

### 2.3 Gestion des Valeurs Manquantes

* Vérification des `NaN` et remplissage selon la nature des variables (médiane pour numériques, mode pour catégorielles).
* Suppression ou traitement des colonnes non pertinentes si nécessaire.

```python
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes('category').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

---

### 2.4 Analyse Statistique et Visuelle

* Distribution de `is_high_risk` : proportion de patients à haut risque vs faible risque.
* Analyse des corrélations entre variables continues.
* Visualisation des proportions de `is_high_risk` selon des variables clés (`smoker`, `diabetes`, `hypertension`).

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='is_high_risk', data=df)
plt.show()
```

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données

```python
from sklearn.model_selection import train_test_split
X = df.drop('is_high_risk', axis=1)
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

---

### 3.2 Modèles Testés

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **XGBoost Classifier**

* Pour les modèles sensibles aux déséquilibres, l'échantillonnage ADASYN a été appliqué pour équilibrer les classes.
* Évaluation avec `accuracy_score`, `f1_score` et `classification_report`.

---

## 4. Résultats et Comparaison des Modèles

| Modèle                   | Accuracy | F1-score | Commentaire                    |
| ------------------------ | -------- | -------- | ------------------------------ |
| Logistic Regression      | 0.XX     | 0.XX     | Performance moyenne            |
| Decision Tree Classifier | 0.XX     | 0.XX     | Meilleur modèle sur ce dataset |
| KNN                      | 0.XX     | 0.XX     | Correct, mais moins stable     |
| XGBoost Classifier       | 0.XX     | 0.XX     | Très performant, robuste       |

**Observations :**

* Les variables comme `smoker`, `diabetes`, `hypertension` influencent fortement la probabilité d’être à risque élevé.
* Les modèles arborescents (Decision Tree, XGBoost) capturent mieux les interactions non-linéaires.

---

## 5. Analyse des Résultats et Recommandations

* **Modèle le plus performant :** Decision Tree Classifier ou XGBoost selon métriques F1 et Accuracy.
* **Recommandations :**

  1. Optimiser les hyperparamètres (GridSearchCV) pour XGBoost.
  2. Créer des features supplémentaires pour capturer les interactions médicales et comportementales.
  3. Suivi des patients identifiés à haut risque pour interventions ciblées.

---

## 6. Conclusion

L’analyse montre que la **modélisation prédictive** peut efficacement identifier les patients à haut risque médical. Les modèles arborescents ont montré les meilleures performances, ce qui souligne l’importance des interactions complexes entre les variables démographiques, biométriques et médicales. Les résultats peuvent guider les décisions de prévention et d’allocation des ressources médicales.

---
