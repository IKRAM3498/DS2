# **IKRAM EL-WALI**
<img width="629" height="635" alt="image" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

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

---

# **Table des Matières**

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Analyse Exploratoire des Données (EDA)](#2-analyse-exploratoire-des-données-eda)

   * [Chargement et Structure](#21-chargement-et-structure)
   * [Prétraitement et Encodage](#22-prétraitement-et-encodage)
   * [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)
   * [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)
4. [Résultats et Performances](#4-résultats-et-performances)
5. [Analyse et Recommandations](#5-analyse-et-recommandations)
6. [Conclusion](#6-conclusion)

---

# **1. Introduction et Contexte**

Ce rapport présente une analyse complète d’un dataset médical provenant de Kaggle visant à prédire les **dépenses annuelles de santé** d’un individu.

L’objectif principal du projet est de :

✔️ comprendre quels facteurs influencent les coûts médicaux
✔️ construire plusieurs modèles de régression
✔️ comparer leurs performances
✔️ identifier le meilleur modèle pour la prédiction des coûts

Nous avons suivi les étapes classiques : EDA, nettoyage, encodage, normalisation, modélisation et interprétation.

---

# **2. Analyse Exploratoire des Données (EDA)**

## **2.1 Chargement et Structure du Dataset**

Le dataset `medical_insurance.csv` contient :

* **100 000 individus**
* **54+ colonnes**

### **Principales catégories de variables :**

1. **Démographie & Socio-économie :**
   age, sex, income, region, education, household_size…

2. **Habitudes de vie :**
   bmi, smoker, alcohol_freq, exercise_frequency…

3. **Santé clinique :**
   hypertension, diabetes, kidney_disease, systolic_bp…

4. **Utilisation médicale :**
   visits_last_year, hospitalizations_last_3yrs…

5. **Assurance :**
   plan_type, deductible, copay…

6. **Coûts et sinistres :**
   annual_medical_cost (variable cible), claims_count…

---

## **2.2 Prétraitement et Encodage**

### **Encodage des variables catégorielles**

* `sex`, `region`, `marital_status`, `employment_status`, `plan_type`, etc.
  ➡️ encodés en **One-Hot Encoding**

### **Suppression des colonnes inutiles**

* `person_id` supprimé (identifiant)

### **Normalisation**

Certaines variables (bmi, bp, ldl…) ont été standardisées pour les modèles sensibles aux échelles.

---

## **2.3 Gestion des Valeurs Manquantes**

Après inspection :

✔️ **Aucune valeur manquante importante**
✔️ Le dataset est propre et directement exploitable.

---

## **2.4 Analyse Statistique et Visuelle**

Les analyses montrent :

* les coûts médicaux varient entre **1100 $** et plus de **1 000 000 $**
* les personnes âgées, fumeuses, diabétiques ou hypertendues dépensent en moyenne plus
* le BMI, la pression artérielle et le nombre de maladies chroniques sont fortement corrélés au coût total

Des heatmaps et histogrammes ont également été générés.

---

# **3. Méthodologie de Modélisation**

Nous avons testé plusieurs modèles de régression :

1. Régression Linéaire
2. Régression Polynomiale
3. Arbre de Décision
4. Forêt Aléatoire
5. Régression SVR

### **Séparation Train/Test**

* **80%** pour l’entraînement
* **20%** pour le test

---

# **4. Résultats et Performances**

| Modèle                 | RMSE                  | R²           |
| ---------------------- | --------------------- | ------------ |
| Régression Linéaire    | élevé                 | faible       |
| Régression Polynomiale | meilleur mais overfit | moyen        |
| Arbre de Décision      | bon                   | bon          |
| **Forêt Aléatoire**    | **excellent**         | **très bon** |
| SVR                    | lent sur 100k lignes  | correct      |

📌 **Le meilleur modèle est la Forêt Aléatoire.**

Elle offre la meilleure précision grâce à sa capacité à capturer les relations non linéaires.

---

# **5. Analyse et Recommandations**

* Les coûts médicaux dépendent largement du **profil clinique** (hypertension, diabète…)
* Le BMI est une variable fortement prédictive
* Les individus avec plus de 3 maladies chroniques ont des coûts très élevés
* Le revenu n’influence pas le coût médical (car les soins ne dépendent pas du salaire)
* Le type de plan d’assurance (deductible, copay) joue un rôle majeur

---

# **6. Conclusion**

L’étude a permis de :

✔️ comprendre profondément les facteurs qui influencent les coûts médicaux
✔️ améliorer la prédiction via plusieurs modèles
✔️ valider que la **Forêt Aléatoire** est le meilleur modèle grâce à sa performance

Ce dataset permet également d’étendre le projet vers :

* la classification du risque
* la détection de fraude
* le scoring assurantiel

---




