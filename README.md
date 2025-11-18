#  Profilage du Trafic Réseau par Clustering K-Means

##  Introduction

Ce projet implémente une solution d'apprentissage non supervisé (Clustering K-Means) pour analyser et profiler le trafic réseau non étiqueté d'une grande organisation. L'objectif est de découvrir des modèles de comportement distincts dans les flux de données (web, streaming, transfert de fichiers, etc.) afin d'améliorer la gestion des ressources et la surveillance de la sécurité.

| Statut | État du Modèle |
| :--- | :--- |
| **Méthode** | K-Means (Apprentissage Non Supervisé) |
| **Jeu de Données** | network_traffic_raw.csv (1825 enregistrements) |
| **Résultat Final** | **6 Clusters** de comportement identifiés (K Optimal) |

***

##  Objectifs Commerciaux

L'analyse a été menée pour répondre aux besoins suivants de l'organisation :
1. Identifier les **modèles distincts** de comportement du trafic réseau.
2. Comprendre l'**utilisation des ressources** par type de trafic.
3. Créer des **lignes de base (baselines)** pour la détection d'anomalies.
4. Optimiser les politiques de **Qualité de Service (QoS)**.

***

## Méthodologie et Préparation des Données (Tâches 2 & 4)

Le succès de K-Means reposant sur la qualité des données, des étapes de prétraitement rigoureuses ont été appliquées :

### 1. Nettoyage et Robustesse
* **Correction des Invalides :** Remplacement des valeurs négatives par `NaN` (impossible logiquement).
* **Gestion des Manquants :** Suppression des lignes `NaN` (317 lignes supprimées, 1508 conservées).
* **Traitement des Outliers :** **Plafonnement (Capping IQR)** des caractéristiques asymétriques (`TotalBytes`, `ByteRate`, etc.) pour neutraliser leur impact sur l'algorithme.

### 2. Standardisation Algorithmique
* **Sélection :** Exclusion de `FlowID`.
* **Encodage :** Conversion de la colonne `Protocol` (TCP/UDP) par **One-Hot Encoding**.
* **Mise à l'Échelle (Scaling) :** Application du **StandardScaler** à toutes les 18 caractéristiques finales. Ceci est CRITIQUE pour assurer que toutes les variables contribuent équitablement à la distance euclidienne de K-Means.

### 3. Détermination du K Optimal (Tâche 5)
L'analyse de la **Méthode du Coude (WCSS)** a montré un point d'inflexion majeur à **K=6**, indiquant six groupes de comportements distincts.

***

## Résultats et Profilage des Clusters (Tâche 8)

Le modèle K-Means a réussi à segmenter le trafic en 6 profils bien séparés.

| Cluster | Nom Proposé | Protocole Dominant | Durée Moyenne | Débit (ByteRate) | Fonction Typique |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C-0** | **Bulk Transfer High-Rate** | TCP (100%) | Moyenne (567 s) | Très Élevé (11.25 Mo/s) | Transferts de gros fichiers avec un débit soutenu. |
| **C-1** | **Interactive Sessions** | TCP (99.3%) | Très Courte (15 s) | Faible (31.7 Ko/s) | Requêtes de base, sessions interactives rapides. |
| **C-2** | **High-Volume Data Transfer** | TCP (100%) | Courte/Moyenne (323 s) | Très Élevé (12.53 Mo/s) | Transferts de volume massif sur une courte période. |
| **C-3** | **DNS / Network Control** | UDP (84.5%) | Très Courte (10 s) | Très Faible (2.56 Ko/s) | Essentiel pour le réseau (DNS, NTP, etc.). |
| **C-4** | **Persistent Application Sessions** | TCP (99.5%) | Très Longue (1834 s) | Modéré (5.30 Mo/s) | Sessions longues (VPN, applications critiques). |
| **C-5** | **IoT / Background Traffic** | UDP (65.9%) | Extrêmement Longue (2739 s) | Très Faible (0.27 Mo/s) | Sondage, mise à jour en arrière-plan, trafic IoT. |

***

##  Recommandations pour la Gestion du Réseau (Tâche 9)

Les clusters fournissent des aperçus actionnables pour les politiques de sécurité et de QoS :

* **Priorité Critique (QoS) :** Les flux du **Cluster C-3 (DNS/Contrôle)** doivent bénéficier de la plus haute priorité et de la plus faible latence pour garantir la stabilité du réseau.
* **Sécurité :** Les clusters **C-3 et C-5** nécessitent une surveillance accrue des anomalies : un volume soudainement élevé de paquets ou d'octets pourrait indiquer un *botnet* (C-3) ou un appareil compromis (C-5).
* **Allocation de Bande Passante :** Les flux de **C-0 et C-2** (Transfert de Volume) peuvent être limités ou planifiés en dehors des heures de pointe pour éviter la congestion des utilisateurs interactifs (C-1 et C-4).

***

##  Structure du Projet et Fichiers

| Nom du Fichier | Rôle | Statut |
| :--- | :--- | :--- |
| `network_traffic_raw.csv` | Jeu de données brut (Input). | Fourni |
| `network_traffic_clean.csv` | Données nettoyées, mais non mises à l'échelle (Tâche 2). | Généré |
| `network_traffic_scaled.csv` | Données finales, mises à l'échelle pour K-Means (Tâche 4). | Généré |
| `network_traffic_clustered.csv` | Données avec la colonne `Cluster` attribuée (Résultat final). | Généré |
| `courbe_du_coude.png` | Visualisation utilisée pour justifier le choix de $K=6$. | Généré |

### Comment Exécuter l'Analyse
1. Placer `network_traffic_raw.csv` dans le répertoire du projet.
2. Installer les dépendances : `pip install pandas numpy scikit-learn matplotlib`.
3. Exécuter le script contenant la Tâche 2, Tâche 4, Tâche 5 et Tâche 6.