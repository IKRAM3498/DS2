#Compte rendu – Analyse de la Qualité du Vin (Vinho Verde)

## **Étudiant** :  IKRAM EL-WALI
## **Classe** : CAC2 



# **1. Introduction et Contexte**
Le présent compte rendu porte sur l’analyse de deux jeux de données réels concernant les vins portugais *Vinho Verde* (rouge et blanc). L’objectif principal est de prédire la **qualité du vin** à partir de ses caractéristiques physico-chimiques.  

Ce travail s’inscrit dans le cadre d’un apprentissage supervisé appliqué à un dataset à la fois **multivarié**, **réel**, et **déséquilibré**, où la variable cible (*quality*) varie de 0 à 10.

---

# **2. Description du Dataset**
Les deux fichiers fournis (`winequality-red.csv` et `winequality-white.csv`) regroupent au total **4898 échantillons** et **11 variables explicatives continues**.

### **Variables principales :**
- Acidité fixe, acidité volatile, acide citrique  
- Sucre résiduel  
- Chlorures  
- Dioxyde de soufre libre / total  
- Densité  
- pH  
- Sulfates  
- Teneur en alcool  

### **Variable cible :**
- **Quality** (note sensorielle de 0 à 10)  

### **Particularités du dataset :**
- Pas de valeurs manquantes.  
- Données déséquilibrées : la majorité des vins ont une note entre 5 et 7.  
- Les variables sont sur **des échelles très différentes**, ce qui influence fortement les modèles basés sur les distances.

---

# **3. Analyse Exploratoire des Données**

### **3.1 Structure générale**
L’analyse de la structure montre un dataset propre, sans valeurs manquantes et avec un grand nombre d’observations.

### **3.2 Distribution de la qualité**
La majorité des vins sont notés « moyens ». Les excellents ou très mauvais vins sont rares, ce qui rend la tâche de classification plus difficile.

### **3.3 Corrélations**
Les variables telles que **l’alcool** et **les sulfates** présentent une corrélation plus élevée avec la qualité. À l’inverse, la densité ou l’acidité fixe ont un impact plus faible.

### **3.4 Variation d’échelle**
Certaines variables (ex : SO₂ total) ont des valeurs beaucoup plus élevées que d’autres (ex : chlorures, pH). Cela rend nécessaire la **normalisation** avant toute modélisation métrique.

---

# **4. Méthodologie de Modélisation**
L’objectif est de prédire la note de qualité, considérée ici comme un **problème de régression**.

### **4.1 Séparation des données**
Les données ont été divisées en :
- **Entraînement :** 80%  
- **Test :** 20%  

### **4.2 Normalisation**
Étant donné l’hétérogénéité d’échelle entre les variables, un **StandardScaler** a été appliqué pour centrer (0) et réduire (1) toutes les caractéristiques.

### **4.3 Modèle choisi : Random Forest Regressor**
Le modèle utilisé est une **forêt aléatoire**, car :
- elle gère bien les variables continues,  
- elle est robuste au bruit,  
- elle indique l’importance des features.  

Le modèle contient **200 arbres** et utilise la moyenne pour prédire la qualité.

---

# **5. Résultats et Interprétation**

### **5.1 Performance globale**
Les métriques principales obtenues sont :
- **MSE (Mean Squared Error)** : erreur quadratique moyenne du modèle.  
- **R² (Coefficient de détermination)** : pourcentage de variance expliquée.

Un R² élevé indique que le modèle prédit correctement la qualité du vin à partir des données physico-chimiques.

### **5.2 Importance des caractéristiques**
Le modèle permet d’identifier les variables ayant le plus d’influence sur la qualité du vin :
- **Teneur en alcool** : la plus corrélée positivement avec la qualité.  
- **Sulfates** : liés à la stabilité du vin.  
- **Acidité volatile** : fortement négative si trop élevée.  

Ces résultats sont cohérents avec les observations oenologiques réelles.

### **5.3 Sauvegarde et visualisation**
Les importances des features ont été enregistrées dans un fichier `.csv` et visualisées sous forme de diagramme.

---

# **6. Conclusion**
L’étude menée sur le dataset *Wine Quality* montre qu’il est possible d’obtenir un modèle prédictif fiable à partir de simples analyses physico-chimiques. Les principaux enseignements sont :

1. **La normalisation est indispensable**, car les variables ont des échelles très différentes.
2. Le modèle **Random Forest** donne de bonnes performances sur ce type de données.
3. Les variables **alcool**, **sulfates** et **acidité volatile** sont les facteurs majeurs influençant la qualité.
4. Un modèle de régression bien construit peut servir d’outil d’aide à la décision dans l’industrie viticole.

Cette analyse confirme l’importance du prétraitement des données, de la sélection de modèle et de l’interprétation des résultats dans un projet de Data Science complet.
