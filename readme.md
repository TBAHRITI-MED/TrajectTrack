# 🗺️ Application de Suivi de Trajectoire et Analyse de Données

![Bannière du projet](https://github.com/user/repo/banner.png)

## 📋 Vue d'ensemble

Cette application offre une solution complète pour la collecte, le stockage, la visualisation et l'analyse avancée de données de trajectoire. Elle se compose de trois composantes interconnectées :

- 📱 **Application mobile iOS** (Swift) - Collecte de données GPS en temps réel
- 🌐 **API REST** (Flask) et **Base de données** (PostgreSQL sur Render) - Stockage centralisé
- 📊 **Application web** (Streamlit) - Visualisation et analyse avancée

## ✨ Fonctionnalités principales

### 📱 Application mobile
- 📍 Capture précise des coordonnées GPS et calcul de la vitesse
- 🔋 Optimisation de la consommation de batterie
- 🔄 Synchronisation avec le serveur et gestion de la connectivité intermittente
- 📊 Visualisation basique des données en temps réel

### 🖥️ Application web
- 🗺️ Visualisation interactive des trajectoires sur des cartes
- 📈 Analyses statistiques détaillées (vitesse, distance, durée)
- 🔍 Détection d'anomalies via des règles et machine learning
- 🧩 Clustering pour identifier des zones d'intérêt (DBSCAN, K-Means)
- 🌟 Algorithme innovant de voisinage itératif
- 📝 Outils d'analyse de segments personnalisés

### 🔄 Intégration backend
- 🔐 API REST sécurisée pour la transmission des données
- 💾 Base de données PostgreSQL hébergée sur Render
- 🧠 Entraînement et persistance des modèles ML
- 📈 Calcul de statistiques et de métriques avancées

## 📸 Captures d'écran

<div align="center">
  <img src="screenshots/mobile-app.png" alt="Application mobile" width="200"/>
  <img src="screenshots/web-dashboard.png" alt="Dashboard web" width="600"/>
  <img src="screenshots/trajectory-analysis.png" alt="Analyse de trajectoire" width="400"/>
  <img src="screenshots/clustering.png" alt="Clustering" width="400"/>
</div>

## 🧪 Technologies utilisées

### 📱 Application mobile
- Swift 5
- CoreLocation
- URLSession
- SwiftUI

### 🌐 Backend
- Python 3.8+
- Flask
- PostgreSQL
- SQLAlchemy
- JWT

### 📊 Frontend
- Streamlit
- Folium
- Plotly
- Pandas
- NumPy
- Scikit-learn

## 🏗️ Architecture du système

```
┌─────────────────┐         ┌────────────────┐         ┌───────────────────┐
│  Application    │         │                │         │   Application      │
│  Mobile Swift   │─────────▶  API Flask     │─────────▶   Web Streamlit    │
│  (iOS)          │         │  + PostgreSQL  │         │   (Analyse)        │
└─────────────────┘         └────────────────┘         └───────────────────┘
      Collecte                   Stockage                   Visualisation
      de données                 centralisé                  et analyse
```

## 📋 Structure du projet

### 📱 Application mobile
```
TrajectoryTracker/
├── Models/                  # Modèles de données
├── ViewModels/              # Logique métier
├── Views/                   # Interface utilisateur SwiftUI
├── Services/                # Services (localisation, API, etc.)
├── Utilities/               # Utilitaires et helpers
└── Resources/               # Ressources (images, etc.)
```

### 🖥️ Application web
```
dash_app.py                  # Application principale Streamlit
├── Configuration/           # Initialisation et configuration
├── Fonctions utilitaires/   # Calculs génériques et visualisations
├── Fonctions d'analyse ML/  # Algorithmes ML (clustering, anomalies)
├── Module clustering avancé/# Algorithmes avancés et voisinage itératif
├── Gestion des données/     # Accès et manipulation des données
└── Interface utilisateur/   # Pages et composants Streamlit
```

## 🧠 Algorithmes et innovations techniques

### 🔍 Détection d'anomalies
- **Isolation Forest** pour identifier les points statistiquement aberrants
- **Règles métier** pour détecter des vitesses anormalement hautes ou basses

### 🧩 Clustering
- **DBSCAN** optimisé pour les données géospatiales
- **K-Means** pour le partitionnement en groupes homogènes

### 🌟 Voisinage itératif
Notre innovation majeure, un algorithme qui construit progressivement un voisinage en suivant les structures naturelles des données, avec trois méthodes de liaison :
- **Simple** : distance minimale au voisinage
- **Complète** : distance maximale au voisinage
- **Moyenne** : distance moyenne au voisinage

## 🔗 Liens

- [Documentation API](https://github.com/yourusername/trajectory-api/docs)
- [Wiki du projet](https://github.com/yourusername/trajectory-dashboard/wiki)
- [Rapport de projet complet](https://link-to-report.pdf)

---

<div align="center">
  <p>Développé avec ❤️ à l'Université de Reims Champagne-Ardenne</p>
  <p>Master Intelligence Artificielle - 2025</p>
</div>