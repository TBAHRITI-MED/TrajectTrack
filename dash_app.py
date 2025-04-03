# dash_app.py - Application Streamlit corrigée

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime, timedelta
from scipy.spatial import ConvexHull
from geopy.distance import geodesic
import time
import threading
import math
import json  # Ajout de l'import json
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
# Supprimez la duplication
import folium
from streamlit_folium import folium_static, st_folium
from folium.plugins import HeatMap, MarkerCluster
from flask import Flask, request, jsonify
import threading
from flask_cors import CORS

# Dans la section où vous créez l'application Flask
flask_app = Flask(__name__)
CORS(flask_app, resources={r"/api/*": {"origins": "*"}}) 


@flask_app.route("/api/push_data", methods=["POST"])
def api_push_data():
    print("Requête reçue:", request.headers)
    print("Données:", request.get_data())
    if not request.json:
        return jsonify({"error": "No JSON body"}), 400

    body = request.json

    # Si les données sont envoyées sous la clé "data", décoder la chaîne JSON imbriquée
    if "data" in body:
        try:
            if isinstance(body["data"], str):
                body = json.loads(body["data"])
            else:
                body = body["data"]
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        lat = float(body.get("latitude", 0))
        lon = float(body.get("longitude", 0))
        speed = float(body.get("speed", 0))
    except ValueError:
        return jsonify({"error": "Invalid numeric values"}), 400

    # Ajouter la donnée avec votre fonction push_data existante
    result = push_data(lat, lon, speed)
    
    return jsonify(result), 200

# Fonction pour démarrer le serveur Flask dans un thread séparé
def run_flask_server():
    # Configurer le port (différent de celui de Streamlit)
    port = int(os.environ.get("FLASK_PORT", 5001))
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
# Configuration de la page
st.set_page_config(
    page_title="Analyse de Données Spatiales",
    page_icon="🗺️",
    layout="wide"
)

# ---------------------------------------------------
# 1. Configuration de la base de données
# ---------------------------------------------------
# Vous pouvez intégrer directement votre base de données dans Streamlit
@st.cache_resource

def init_db():
    # Utiliser l'URL externe de la base de données PostgreSQL de Render
    db_url = "postgresql://data_suivi_user:oYFlVBF6UAaRZk3el2vtXhVPtvOn9uzW@dpg-cuue8c52ng1s739p7grg-a.oregon-postgres.render.com/data_suivi"
    
    engine = create_engine(db_url)
    Base = declarative_base()
    
    # Définir le modèle SensorData
    class SensorData(Base):
        __tablename__ = 'sensor_data'
        id = Column(Integer, primary_key=True)
        latitude = Column(Float, nullable=False)
        longitude = Column(Float, nullable=False)
        speed = Column(Float, nullable=False)
        # Pas de colonne timestamp pour éviter l'erreur précédente
    
    # Créer les tables si elles n'existent pas
    Base.metadata.create_all(engine)
    
    # Créer une session
    Session = sessionmaker(bind=engine)
    
    return {"engine": engine, "Session": Session, "SensorData": SensorData, "func": func}

# Initialiser la base de données
db = init_db()
SensorData = db.get("SensorData")
Session = db.get("Session")

# ---------------------------------------------------
# 2. Variables globales pour l'analyse
# ---------------------------------------------------
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0

if 'current_explanation' not in st.session_state:
    st.session_state.current_explanation = "Pas d'analyse disponible."

if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False

if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {
        'isolation_forest': None,
        'dbscan': None,
        'is_trained': False
    }

# Initialiser la session state pour les points sélectionnés
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []

# ---------------------------------------------------
# 3. Fonctions utilitaires
# ---------------------------------------------------


# Fonction pour calculer la distance entre deux points en mètres
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points en mètres"""
    # Convertir en radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Formule de haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Rayon de la Terre en km
    return km * 1000  # En mètres

# Fonction pour calculer la distance totale du parcours
def calculate_distance(df):
    """Calcule la distance totale parcourue en km"""
    if len(df) < 2:
        return 0
    
    total_distance = 0
    for i in range(1, len(df)):
        # Calcul de la distance entre deux points consécutifs
        lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
        lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
        
        # Conversion en radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Formule de haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c  # Rayon de la Terre en km
        
        total_distance += distance
    
    return total_distance

# Fonction pour calculer la durée
def calculate_duration(df):
    """Calcule la durée totale entre le premier et le dernier point"""
    if len(df) < 2 or 'timestamp' not in df.columns:
        return "Non disponible"
    
    # S'assurer que timestamp est bien un datetime
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculer la différence
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    duration = end_time - start_time
    
    # Formater la durée en heures:minutes:secondes
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# Fonction pour déterminer si un point est sur un segment
def is_point_on_segment(lat_p, lon_p, lat_a, lon_a, lat_b, lon_b, width=30):
    """
    Détermine si un point P(lat_p, lon_p) est sur un segment AB, à une distance maximale de 'width' mètres.
    
    Args:
        lat_p, lon_p: Coordonnées du point à tester
        lat_a, lon_a: Coordonnées du point A du segment
        lat_b, lon_b: Coordonnées du point B du segment
        width: Distance maximale en mètres du point au segment
        
    Returns:
        bool: True si le point est sur le segment à la distance spécifiée, False sinon
    """
    # Calculer la distance du point à la ligne
    def distance_point_to_line(p, a, b):
        # Convertir en coordonnées cartésiennes simplifiées
        R = 111320.0  # Mètres par degré approximativement
        ax = a[1] * R * math.cos(math.radians(a[0]))
        ay = a[0] * R
        bx = b[1] * R * math.cos(math.radians(b[0]))
        by = b[0] * R
        px = p[1] * R * math.cos(math.radians(p[0]))
        py = p[0] * R
        
        # Vecteur AB
        abx = bx - ax
        aby = by - ay
        
        # Vecteur AP
        apx = px - ax
        apy = py - ay
        
        # Projection de AP sur AB
        ab_length_squared = abx**2 + aby**2
        
        if ab_length_squared == 0:  # A = B
            return math.sqrt((px-ax)**2 + (py-ay)**2)
        
        t = max(0, min(1, (apx*abx + apy*aby) / ab_length_squared))
        
        # Point de projection
        projx = ax + t * abx
        projy = ay + t * aby
        
        # Distance du point à la projection
        return math.sqrt((px-projx)**2 + (py-projy)**2)
    
    dist = distance_point_to_line([lat_p, lon_p], [lat_a, lon_a], [lat_b, lon_b])
    return dist <= width

# ---------------------------------------------------
# 4. Fonctions d'analyse ML
# ---------------------------------------------------
def train_ml_models():
    """Entraîne les modèles ML avec les données existantes"""
    try:
        # Récupérer toutes les données
        session = Session()
        points = session.query(SensorData).all()
        
        if len(points) < 10:
            st.warning("⚠️ Pas assez de données pour l'entraînement (minimum 10 points)")
            return False
            
        # Préparer les données pour le clustering géographique
        coords = np.array([[p.latitude, p.longitude] for p in points])
        speeds = np.array([[p.speed] for p in points])
        
        # DBSCAN pour le clustering des zones
        eps_meters = 50  # 50 mètres entre points pour former un cluster
        kms_per_radian = 6371.0088  # Rayon de la Terre en km
        epsilon = eps_meters / 1000 / kms_per_radian
        
        # Entraîner DBSCAN
        dbscan = DBSCAN(
            eps=epsilon, 
            min_samples=5, 
            algorithm='ball_tree',
            metric='haversine'
        )
        dbscan.fit(np.radians(coords))
        
        # Isolation Forest pour détecter les anomalies de vitesse
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(speeds)
        
        # Sauvegarder les modèles dans la session state
        st.session_state.ml_models['isolation_forest'] = iso_forest
        st.session_state.ml_models['dbscan'] = dbscan
        st.session_state.ml_models['is_trained'] = True
        
        # Sauvegarder sur disque (optionnel)
        model_dir = "ml_models"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(iso_forest, os.path.join(model_dir, "iso_forest.pkl"))
        joblib.dump(dbscan, os.path.join(model_dir, "dbscan.pkl"))
        
        st.success(f"✅ Modèles ML entraînés avec succès sur {len(points)} points")
        return True
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'entraînement ML: {e}")
        return False
    finally:
        session.close()
def load_ml_models():
    """Tente de charger les modèles préexistants"""
    try:
        model_dir = "ml_models"
        iso_path = os.path.join(model_dir, "iso_forest.pkl")
        dbscan_path = os.path.join(model_dir, "dbscan.pkl")
        
        if os.path.exists(iso_path) and os.path.exists(dbscan_path):
            st.session_state.ml_models['isolation_forest'] = joblib.load(iso_path)
            st.session_state.ml_models['dbscan'] = joblib.load(dbscan_path)
            st.session_state.ml_models['is_trained'] = True
            return True
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les modèles ML: {e}")
    
    return False
def analyze_point(lat, lon, speed):
    """Analyse un point avec les modèles ML"""
    if not st.session_state.ml_models['is_trained']:
        if not load_ml_models():
            # Tenter un entraînement si pas de modèles préexistants
            train_ml_models()
        
        if not st.session_state.ml_models['is_trained']:
            return {
                "status": "no_model",
                "message": "Aucun modèle ML disponible"
            }
    
    result = {"status": "success"}
    
    # Analyser la vitesse avec Isolation Forest
    if st.session_state.ml_models['isolation_forest'] is not None:
        # Prédiction d'anomalie
        speed_array = np.array([[speed]])
        is_anomaly = st.session_state.ml_models['isolation_forest'].predict(speed_array)[0] == -1
        anomaly_score = st.session_state.ml_models['isolation_forest'].score_samples(speed_array)[0]
        
        result["speed_analysis"] = {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "message": "Vitesse anormale détectée" if is_anomaly else "Vitesse normale"
        }
    
    # Trouver les zones populaires
    session = Session()
    points = session.query(SensorData).all()
    if len(points) >= 10:
        clusters = {}
        coords = np.array([[p.latitude, p.longitude] for p in points])
        
        if st.session_state.ml_models['dbscan'] is not None:
            # Appliquer les labels de cluster à tous les points
            labels = st.session_state.ml_models['dbscan'].labels_
            
            # Compter les points par cluster et trouver les centres
            for i, label in enumerate(labels):
                if label != -1:  # Ignorer les points considérés comme du bruit
                    if label not in clusters:
                        clusters[label] = {
                            "count": 0,
                            "lat_sum": 0,
                            "lon_sum": 0,
                            "points": []
                        }
                    clusters[label]["count"] += 1
                    clusters[label]["lat_sum"] += coords[i][0]
                    clusters[label]["lon_sum"] += coords[i][1]
                    clusters[label]["points"].append(coords[i])
            
            # Calculer les centres de cluster
            popular_zones = []
            for label, data in clusters.items():
                if data["count"] >= 5:  # Au moins 5 points
                    center_lat = data["lat_sum"] / data["count"]
                    center_lon = data["lon_sum"] / data["count"]
                    popular_zones.append({
                        "cluster_id": int(label),
                        "center": [float(center_lat), float(center_lon)],
                        "point_count": data["count"]
                    })
            
            # Trier par nombre de points
            popular_zones.sort(key=lambda x: x["point_count"], reverse=True)
            
            # Garder les 3 premiers
            result["popular_zones"] = popular_zones[:3]
            
            # Vérifier si le point actuel est dans une zone populaire
            is_in_zone = False
            for zone in popular_zones[:3]:
                # Distance approximative
                dist = haversine_distance(lat, lon, zone["center"][0], zone["center"][1])
                if dist < 50:  # 50 mètres
                    is_in_zone = True
                    result["current_zone"] = {
                        "in_popular_zone": True,
                        "zone_rank": popular_zones.index(zone) + 1,
                        "cluster_id": zone["cluster_id"],
                        "point_count": zone["point_count"]
                    }
                    break
            
            if not is_in_zone:
                result["current_zone"] = {"in_popular_zone": False}
    
    session.close()
    return result
def analyser_ralentissement_async(speed, avg_speed):
    """
    Fonction asynchrone qui analyse le ralentissement avec OpenAI
    """
    try:
        st.session_state.analysis_in_progress = True
        
        # Vérifier si une clé API OpenAI est disponible
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.session_state.current_explanation = "Clé API OpenAI non disponible"
            return
        
        # Initialiser le client OpenAI
        client = OpenAI(api_key=api_key)
        
        # Appel API avec un timeout court et un prompt simplifié
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "Tu es un expert trafic. Réponse courte (30 mots max)."},
                {"role": "user", "content": f"Pourquoi vitesse={speed}m/s vs moyenne={avg_speed:.2f}m/s? Raison courte."}
            ],
            max_tokens=50
        )
        
        # Extraction de la réponse
        if hasattr(response, 'choices') and response.choices:
            result = response.choices[0].message.content
            st.session_state.current_explanation = result
        else:
            st.session_state.current_explanation = "Aucune explication trouvée."
            
    except Exception as e:
        st.session_state.current_explanation = f"Erreur d'analyse: {str(e)}"
    finally:
        st.session_state.analysis_in_progress = False
def display_anomaly_results(df_anomalies):
                # Création de la carte avec anomalies
                m_anomalies = folium.Map(
                    location=[df_anomalies['latitude'].mean(), df_anomalies['longitude'].mean()], 
                    zoom_start=14
                )
                
                # Taille des points
                point_size = 5
                
                # Tracer les points normaux
                for idx, row in df_anomalies[df_anomalies['anomaly'] == 1].iterrows():
                    folium.CircleMarker(
                        [row['latitude'], row['longitude']],
                        radius=point_size,
                        color='blue',
                        fill=True,
                        fill_opacity=0.5,
                        popup=f"Normal: {row['speed']:.2f} m/s"
                    ).add_to(m_anomalies)
                
                # Tracer les anomalies avec taille proportionnelle à l'ampleur de l'anomalie
                for idx, row in df_anomalies[df_anomalies['anomaly'] == -1].iterrows():
                    # Calculer la taille en fonction du score d'anomalie
                    # Plus le score est bas, plus l'anomalie est forte
                    score = row['anomaly_score']
                    # Normalisation: plus le score est négatif, plus l'anomalie est importante
                    normalized_score = abs(min(score, 0))
                    # Taille variable: entre taille de base et taille de base + 5
                    radius = point_size + (normalized_score * 5)
                    
                    # Déterminer la couleur en fonction de la vitesse
                    if row['speed'] <= 0.5:  # Vitesse très basse
                        color = 'purple'
                        reason = "Arrêt ou vitesse très basse"
                    elif row['speed'] >= 15:  # Vitesse très élevée
                        color = 'red'
                        reason = "Vitesse très élevée"
                    else:  # Autres anomalies
                        color = 'orange'
                        reason = "Anomalie de comportement"
                    
                    folium.CircleMarker(
                        [row['latitude'], row['longitude']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"Anomalie: {row['speed']:.2f} m/s<br>Score: {row['anomaly_score']:.3f}<br>Raison: {reason}"
                    ).add_to(m_anomalies)
                
                # Afficher la carte avec anomalies
                st.subheader("Carte des anomalies détectées")
                folium_static(m_anomalies, width=800, height=500)
                
                # Statistiques des anomalies
                n_anomalies = (df_anomalies['anomaly'] == -1).sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre d'anomalies", n_anomalies)
                col2.metric("Pourcentage d'anomalies", f"{(n_anomalies / len(df_anomalies) * 100):.1f}%")
                
                # Si on a des vitesses très faibles ou très élevées
                n_low_speed = ((df_anomalies['anomaly'] == -1) & (df_anomalies['speed'] <= 0.5)).sum()
                n_high_speed = ((df_anomalies['anomaly'] == -1) & (df_anomalies['speed'] >= 15)).sum()
                
                col3.metric("Anomalies de vitesse", f"{n_low_speed} basses, {n_high_speed} élevées")
                
                # Distribution des scores d'anomalies
                fig = px.histogram(
                    df_anomalies, 
                    x='anomaly_score', 
                    color='anomaly',
                    color_discrete_map={1: 'blue', -1: 'red'},
                    labels={"anomaly_score": "Score d'anomalie", "count": "Nombre de points"},
                    title="Distribution des scores d'anomalie"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des anomalies détectées
                if n_anomalies > 0:
                    st.subheader("Liste des anomalies détectées")
                    
                    # Ajouter une description de l'anomalie
                    anomalies_df = df_anomalies[df_anomalies['anomaly'] == -1].copy()
                    
                    # Classer les anomalies
                    def classify_anomaly(row):
                        if row['speed'] <= 0.5:
                            return "Arrêt ou vitesse très basse"
                        elif row['speed'] >= 15:
                            return "Vitesse très élevée"
                        else:
                            return "Comportement atypique"
                    
                    anomalies_df['type_anomalie'] = anomalies_df.apply(classify_anomaly, axis=1)
                    
                    # Afficher le tableau des anomalies
                    anomalies_display = anomalies_df[['latitude', 'longitude', 'speed', 'anomaly_score', 'type_anomalie']]
                    anomalies_display = anomalies_display.sort_values(by='anomaly_score')
                    st.dataframe(anomalies_display)
# ---------------------------------------------------
# 5. Fonctions de gestion des données
# ---------------------------------------------------
# Fonction pour afficher les résultats des anomalies

def push_data(lat, lon, speed):
    """Ajoute une nouvelle donnée de capteur et analyse les ralentissements"""
    try:
        session = Session()
        
        # Calculer la vitesse moyenne actuelle
        avg_speed = session.query(func.avg(SensorData.speed)).scalar() or 0
        
        # Ajouter les données
        new_data = SensorData(
            latitude=lat,
            longitude=lon,
            speed=speed
        )
        session.add(new_data)
        session.commit()
        
        # Détection d'un ralentissement
        ralentissement = False
        if avg_speed > 0 and speed < avg_speed * 0.8:
            ralentissement = True
            
            # Vérifier si il faut lancer une analyse
            current_time = time.time()
            if not st.session_state.analysis_in_progress and (current_time - st.session_state.last_analysis_time) > 60:
                st.session_state.last_analysis_time = current_time
                
                # Lancer l'analyse dans un thread séparé
                thread = threading.Thread(
                    target=analyser_ralentissement_async, 
                    args=(speed, avg_speed)
                )
                thread.daemon = True
                thread.start()
        
        return {
            "status": "success",
            "current_speed": speed,
            "average_speed": avg_speed,
            "slowdown_detected": ralentissement
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

def get_all_points():
    """Récupère tous les points de données"""
    try:
        session = Session()
        points = session.query(SensorData).all()
        data = []
        
        for p in points:
            point_data = {
                "id": p.id,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "speed": p.speed
            }
            # Ajouter timestamp seulement s'il existe
            if hasattr(p, 'timestamp') and p.timestamp:
                point_data["timestamp"] = p.timestamp.isoformat()
            
            data.append(point_data)
        
        return {"status": "success", "points": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

def get_latest_point():
    """Récupère le dernier point enregistré"""
    try:
        session = Session()
        latest = session.query(SensorData).order_by(SensorData.id.desc()).first()
        
        if latest:
            # Préparer les données du point
            point_data = {
                "id": latest.id,
                "latitude": latest.latitude,
                "longitude": latest.longitude,
                "speed": latest.speed,
                "is_latest": True
            }
            
            # Ajouter timestamp seulement s'il existe
            if hasattr(latest, 'timestamp') and latest.timestamp:
                point_data["timestamp"] = latest.timestamp.isoformat()
            
            # Obtenir les résultats ML
            ml_results = analyze_point(latest.latitude, latest.longitude, latest.speed)
            
            return {
                "status": "success",
                "latest_point": point_data,
                "ml_analysis": ml_results
            }
        else:
            return {
                "status": "error",
                "message": "Aucun point trouvé"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

# --- Interface utilisateur Streamlit ---

# Titre de l'application
st.title("Tableau de bord d'analyse de données spatiales")

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisir une page",
    ["Carte principale", "Analyse de vitesse", "Clusters", "Anomalies", "Statistiques", "Configuration"]
)
# Charger les données
@st.cache_data(ttl=10)  # Cache pendant 10 secondes
def load_data():
    data = get_all_points()
    if data["status"] == "success":
        return pd.DataFrame(data["points"])
    else:
        st.error(f"Erreur de chargement des données: {data.get('message', 'Erreur inconnue')}")
        return pd.DataFrame()

df = load_data()

# Condition pour afficher une aide si aucune donnée
if len(df) == 0:
    st.warning("Aucune donnée disponible. Utilisez l'onglet Configuration pour ajouter des points ou importer des données.")
    if page != "Configuration":
        st.info("Allez dans la page Configuration pour commencer.")

# Fonction pour créer une carte avec points
def create_map_with_points(df, center=None, zoom=14):
    if len(df) == 0:
        # Coordonnées par défaut (Reims)
        center = [49.2484420, 4.0415017]
        m = folium.Map(location=center, zoom_start=zoom)
        return m
    
    if center is None:
        # Utiliser le centre des données
        center = [df['latitude'].mean(), df['longitude'].mean()]
    
    m = folium.Map(location=center, zoom_start=zoom)
    
    # Ajouter tous les points en gris
    for idx, row in df.iterrows():
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=4,
            color='gray',
            fill=True,
            fill_opacity=0.7,
            popup=f"Point {idx}: {row['speed']:.2f} m/s"
        ).add_to(m)
    
    # Ajouter le dernier point en bleu et plus gros s'il existe
    if len(df) > 0:
        latest = df.iloc[-1]
        folium.CircleMarker(
            [latest['latitude'], latest['longitude']],
            radius=8,
            color='#4285F4',
            fill=True,
            fill_opacity=0.8,
            weight=2,
            popup=f"Dernier point: {latest['speed']:.2f} m/s"
        ).add_to(m)
    
    return m

# -- PAGES --
# Remplacez toutes les occurrences de st.experimental_rerun() par st.rerun()
# Par exemple, dans le code de la page "Carte principale" :

# Modification pour la section de la carte principale pour:
# 1. Permettre de sélectionner les points existants de la trajectoire
# 2. Améliorer l'analyse des segments

if page == "Carte principale":
    st.header("Carte interactive")
    
    # Disposition en colonnes
    col1, col2 = st.columns([3, 1])
    
    # Variable pour stocker temporairement le point sélectionné
    selected_lat, selected_lng = None, None
    selected_existing_point = False
    
    with col1:
        # Créer une carte avec possibilité de clic
        center = [df['latitude'].mean(), df['longitude'].mean()] if len(df) > 0 else [49.2484420, 4.0415017]
        
        # Créer une carte Folium
        m = folium.Map(location=center, zoom_start=14)
        
        # Ajouter les points de données existants avec des popups cliquables
        if len(df) > 0:
            for idx, row in df.iterrows():
                popup_content = f"""
                <strong>Point {idx}</strong><br>
                Vitesse: {row['speed']:.2f} m/s<br>
                <button onclick="
                    const lat = {row['latitude']};
                    const lng = {row['longitude']};
                    document.getElementById('selected-point-info').innerText = 'Point de trajectoire ' + {idx} + ' sélectionné';
                    // Simuler un clic sur la carte pour envoyer les coordonnées
                    var event = new CustomEvent('trajectory_point_selected', {{ 
                        detail: {{ lat: lat, lng: lng, id: {idx} }}
                    }});
                    document.dispatchEvent(event);
                ">Sélectionner ce point</button>
                """
                
                folium.CircleMarker(
                    [row['latitude'], row['longitude']],
                    radius=4,
                    color='gray',
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=200)
                ).add_to(m)
        
        # Ajouter les points déjà sélectionnés sur la carte avec numérotation
        for i, (lat, lon) in enumerate(st.session_state.selected_points):
            folium.Marker(
                [lat, lon], 
                popup=f"Point de segment {i+1}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
        
        # Ajouter les segments entre les points consécutifs
        if len(st.session_state.selected_points) > 1:
            for i in range(1, len(st.session_state.selected_points)):
                start_point = st.session_state.selected_points[i-1]
                end_point = st.session_state.selected_points[i]
                
                folium.PolyLine(
                    [start_point, end_point],
                    color="blue",
                    weight=3,
                    opacity=0.7,
                    popup=f"Segment {i}"
                ).add_to(m)
        
        # Désactiver le cache pour cette carte
        st.cache_data.clear()
        
        # Ajouter un gestionnaire d'événements personnalisé avec JavaScript
        js = """
        <script>
        // Écouter l'événement personnalisé de sélection de point de trajectoire
        document.addEventListener('trajectory_point_selected', function(e) {
            // Créer un élément caché pour stocker les coordonnées
            var hiddenInput = document.createElement('input');
            hiddenInput.type = 'hidden';
            hiddenInput.id = 'selected-trajectory-point';
            hiddenInput.value = JSON.stringify({
                lat: e.detail.lat, 
                lng: e.detail.lng,
                id: e.detail.id
            });
            document.body.appendChild(hiddenInput);
        });
        </script>
        
        <div id="selected-point-info" style="color: blue;"></div>
        """
        
        # Injecter le JavaScript dans la carte
        m.get_root().html.add_child(folium.Element(js))
        
        # Utiliser st_folium pour afficher la carte et récupérer les informations de clic
        map_data = st_folium(
            m, 
            width=800, 
            height=500,
            returned_objects=["last_clicked"],
            use_container_width=True,
            key="map_main"
        )
        
        # Vérifier si un clic a été effectué sur la carte
        if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
            clicked = map_data['last_clicked']
            selected_lat, selected_lng = clicked['lat'], clicked['lng']
            
            # Vérifier si le point est proche d'un point de trajectoire existant
            if len(df) > 0:
                for idx, row in df.iterrows():
                    dist = haversine_distance(
                        selected_lat, selected_lng, 
                        row['latitude'], row['longitude']
                    )
                    # Si le clic est à moins de 10 mètres d'un point existant, utiliser ce point
                    if dist < 10:  # 10 mètres de tolérance
                        selected_lat = row['latitude']
                        selected_lng = row['longitude']
                        selected_existing_point = True
                        st.success(f"✅ Point de trajectoire {idx} sélectionné (distance: {dist:.1f}m)")
                        break
            
            if not selected_existing_point:
                st.success(f"📍 Nouveau point sélectionné: Lat {selected_lat:.6f}, Lng {selected_lng:.6f}")
    
    with col2:
        # Métriques principales
        st.subheader("Informations générales")
        
        if len(df) > 0:
            latest = df.iloc[-1]
            st.metric("Points enregistrés", len(df))
            st.metric("Dernière vitesse", f"{latest['speed']:.2f} m/s")
            
            # Calcul de la vitesse moyenne
            avg_speed = df['speed'].mean()
            st.metric("Vitesse moyenne", f"{avg_speed:.2f} m/s")
            
            # Afficher l'explication IA si disponible
            if st.session_state.current_explanation != "Pas d'analyse disponible.":
                st.info(f"**Explication IA:** {st.session_state.current_explanation}")
            
            # Ajouter un indicateur de statut ML
            if st.session_state.ml_models['is_trained']:
                st.success("Modèles ML: Entraînés")
            else:
                st.warning("Modèles ML: Non entraînés")
        else:
            st.info("Aucune donnée disponible")
        
        # Instructions pour l'utilisateur
        st.info("📌 **Instructions**: Cliquez sur la carte ou sur un point de trajectoire existant pour le sélectionner, puis ajoutez-le à votre segment")
        
        # Afficher les coordonnées du point sélectionné et proposer d'ajouter ce point
        if selected_lat is not None and selected_lng is not None:
            # Bouton pour ajouter le point au segment
            if st.button("Ajouter le point sélectionné au segment"):
                st.session_state.selected_points.append((selected_lat, selected_lng))
                st.success(f"Point {len(st.session_state.selected_points)} ajouté!")
                st.rerun()
        
        # Afficher les points actuellement sélectionnés
        if st.session_state.selected_points:
            st.subheader(f"Points du segment ({len(st.session_state.selected_points)})")
            for i, (lat, lon) in enumerate(st.session_state.selected_points):
                st.text(f"Point {i+1}: Lat {lat:.6f}, Lng {lon:.6f}")
            
            # Boutons pour gérer les points
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Réinitialiser tous les points"):
                    st.session_state.selected_points = []
                    st.success("Points réinitialisés")
                    st.rerun()
            
            with col2:
                if len(st.session_state.selected_points) > 0 and st.button("Supprimer le dernier point"):
                    st.session_state.selected_points.pop()
                    st.success("Dernier point supprimé")
                    st.rerun()
        
        # Ajouter un bouton pour actualiser les données mais uniquement à la demande
        if st.button("Actualiser les données", key="refresh_button"):
            st.cache_data.clear()
            st.rerun()
    
    # Section pour l'analyse de segments
    if len(st.session_state.selected_points) >= 2:
        st.header("Analyse de segments")
        
        # Créer une carte pour visualiser les segments
        m_segments = folium.Map(
            location=[
                sum(p[0] for p in st.session_state.selected_points) / len(st.session_state.selected_points),
                sum(p[1] for p in st.session_state.selected_points) / len(st.session_state.selected_points)
            ],
            zoom_start=14
        )
        
        # Ajouter les points sélectionnés comme marqueurs
        for i, (lat, lon) in enumerate(st.session_state.selected_points):
            folium.Marker(
                [lat, lon],
                popup=f"Point {i+1}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m_segments)
        
        # Calculer et tracer tous les segments
        segment_stats = []
        
        # Tracer les segments entre points consécutifs
        for i in range(1, len(st.session_state.selected_points)):
            start_point = st.session_state.selected_points[i-1]
            end_point = st.session_state.selected_points[i]
            
            # Tracer la ligne
            folium.PolyLine(
                [start_point, end_point],
                color="blue",
                weight=3,
                opacity=0.7,
                popup=f"Segment {i}: Point {i} - Point {i+1}"
            ).add_to(m_segments)
            
            # Calculer la distance pour ce segment
            lat1, lon1 = start_point
            lat2, lon2 = end_point
            
            # Conversion en radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Formule de haversine
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c  # Rayon de la Terre en km
            
            # Ajouter aux statistiques
            segment_stats.append({
                "segment_id": i,
                "from_point": i,
                "to_point": i+1,
                "distance_km": distance,
                "points_on_segment": 0,
                "avg_speed": 0,
                "min_speed": float('inf'),
                "max_speed": 0,
                "points_data": []
            })
            
            # Calculer les points sur ce segment
            segment_width = 30  # mètres
            points_on_segment = []
            
            for idx, row in df.iterrows():
                if is_point_on_segment(
                    row['latitude'], row['longitude'], 
                    start_point[0], start_point[1], 
                    end_point[0], end_point[1], 
                    segment_width
                ):
                    points_on_segment.append(row)
                    # Ajouter le point à la carte avec une couleur spécifique au segment
                    folium.CircleMarker(
                        [row['latitude'], row['longitude']],
                        radius=3,
                        color=f"hsl({(i*50) % 360}, 70%, 50%)",
                        fill=True,
                        fill_opacity=0.5,
                        popup=f"Segment {i}: {row['speed']:.2f} m/s"
                    ).add_to(m_segments)
            
            # Mettre à jour les stats de ce segment
            if points_on_segment:
                speeds = [p['speed'] for p in points_on_segment]
                segment_stats[-1]["points_on_segment"] = len(points_on_segment)
                segment_stats[-1]["avg_speed"] = sum(speeds) / len(speeds)
                segment_stats[-1]["min_speed"] = min(speeds)
                segment_stats[-1]["max_speed"] = max(speeds)
                segment_stats[-1]["points_data"] = points_on_segment
        
        # Ajouter également un segment direct du premier au dernier point (si plus de 2 points)
        if len(st.session_state.selected_points) > 2:
            start_point = st.session_state.selected_points[0]
            end_point = st.session_state.selected_points[-1]
            
            # Tracer la ligne en pointillés
            folium.PolyLine(
                [start_point, end_point],
                color="red",
                weight=2,
                opacity=0.7,
                dash_array="5, 10",
                popup=f"Segment direct: Point 1 - Point {len(st.session_state.selected_points)}"
            ).add_to(m_segments)
            
            # Calculer la distance pour ce segment direct
            lat1, lon1 = start_point
            lat2, lon2 = end_point
            
            # Conversion en radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Formule de haversine
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c  # Rayon de la Terre en km
            
            # Calculer les points sur ce segment direct
            segment_width = 30  # mètres
            points_on_direct = []
            
            for idx, row in df.iterrows():
                if is_point_on_segment(
                    row['latitude'], row['longitude'], 
                    start_point[0], start_point[1], 
                    end_point[0], end_point[1], 
                    segment_width
                ):
                    points_on_direct.append(row)
            
            # Préparer les stats pour le segment direct
            direct_stats = {
                "segment_id": "direct",
                "from_point": 1,
                "to_point": len(st.session_state.selected_points),
                "distance_km": distance,
                "points_on_segment": len(points_on_direct),
                "avg_speed": 0,
                "min_speed": float('inf'),
                "max_speed": 0
            }
            
            # Mettre à jour les stats du segment direct
            if points_on_direct:
                speeds = [p['speed'] for p in points_on_direct]
                direct_stats["avg_speed"] = sum(speeds) / len(speeds)
                direct_stats["min_speed"] = min(speeds)
                direct_stats["max_speed"] = max(speeds)
            
            # Ajouter aux statistiques
            segment_stats.append(direct_stats)
        
        # Afficher la carte
        folium_static(m_segments, width=800, height=500)
        
        # Afficher les statistiques des segments
        st.subheader("Statistiques des segments")
        
        # Créer un DataFrame des statistiques
        stats_columns = ["segment_id", "from_point", "to_point", "distance_km", 
                        "points_on_segment", "avg_speed", "min_speed", "max_speed"]
        stats_df = pd.DataFrame(segment_stats)[stats_columns]
        
        # Formater les chiffres
        for col in ["distance_km", "avg_speed", "min_speed", "max_speed"]:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: round(x, 2))
        
        # Calculer la distance totale des segments consécutifs
        total_distance = sum(stat["distance_km"] for stat in segment_stats if stat["segment_id"] != "direct")
        
        # Afficher les statistiques
        col1, col2, col3 = st.columns(3)
        col1.metric("Distance totale du parcours", f"{total_distance:.2f} km")
        
        if any(stat["points_on_segment"] > 0 for stat in segment_stats if stat["segment_id"] != "direct"):
            avg_speed_all_segments = np.mean([
                stat["avg_speed"] for stat in segment_stats 
                if stat["segment_id"] != "direct" and stat["points_on_segment"] > 0
            ])
            col2.metric("Vitesse moyenne", f"{avg_speed_all_segments:.2f} m/s")
        
        if len(segment_stats) > 0 and "direct" in [str(s["segment_id"]) for s in segment_stats]:
            direct_distance = next(s["distance_km"] for s in segment_stats if s["segment_id"] == "direct")
            col3.metric("Distance directe (vol d'oiseau)", f"{direct_distance:.2f} km")
        
        # Afficher un tableau des statistiques par segment
        st.dataframe(stats_df)
        
        # Graphique comparatif des segments
        st.subheader("Comparaison des segments")
        
        # Filtrer les segments qui ont des points
        segments_with_data = [s for s in segment_stats if s["points_on_segment"] > 0]
        
        if segments_with_data:
            # Comparer les vitesses moyennes des segments
            fig_speed = px.bar(
                x=[f"Segment {s['segment_id']}" if s['segment_id'] != "direct" else "Direct" for s in segments_with_data],
                y=[s["avg_speed"] for s in segments_with_data],
                labels={"x": "Segment", "y": "Vitesse moyenne (m/s)"},
                title="Comparaison des vitesses moyennes par segment",
                color=[
                    "Trajet" if s['segment_id'] != "direct" else "Direct" 
                    for s in segments_with_data
                ],
                color_discrete_map={"Trajet": "blue", "Direct": "red"}
            )
            
            st.plotly_chart(fig_speed, use_container_width=True)
            
            # Comparer la densité de points des segments
            fig_density = px.bar(
                x=[f"Segment {s['segment_id']}" if s['segment_id'] != "direct" else "Direct" for s in segments_with_data],
                y=[s["points_on_segment"] / s["distance_km"] for s in segments_with_data],
                labels={"x": "Segment", "y": "Densité (points/km)"},
                title="Densité de points par segment",
                color=[
                    "Trajet" if s['segment_id'] != "direct" else "Direct" 
                    for s in segments_with_data
                ],
                color_discrete_map={"Trajet": "blue", "Direct": "red"}
            )
            
            st.plotly_chart(fig_density, use_container_width=True)
            
            # Analyse de segment individuel
            st.subheader("Analyse détaillée par segment")
            
            # Liste des segments pour sélection
            segment_options = [f"Segment {s['segment_id']}" if s['segment_id'] != "direct" else "Segment direct" 
                              for s in segment_stats if s["points_on_segment"] > 0]
            
            if segment_options:
                selected_segment = st.selectbox("Sélectionner un segment à analyser", segment_options)
                
                # Trouver l'index du segment sélectionné
                segment_idx = int(selected_segment.split()[-1]) if "direct" not in selected_segment else "direct"
                
                # Trouver les données du segment sélectionné
                selected_segment_data = next((s for s in segment_stats if str(s["segment_id"]) == str(segment_idx)), None)
                
                if selected_segment_data and "points_data" in selected_segment_data and selected_segment_data["points_data"]:
                    # Convertir les points du segment en DataFrame
                    points_df = pd.DataFrame([
                        {
                            "id": p.id if hasattr(p, 'id') else idx,
                            "latitude": p.latitude if hasattr(p, 'latitude') else p['latitude'],
                            "longitude": p.longitude if hasattr(p, 'longitude') else p['longitude'],
                            "speed": p.speed if hasattr(p, 'speed') else p['speed']
                        }
                        for idx, p in enumerate(selected_segment_data["points_data"])
                    ])
                    
                    # Graphique d'évolution de la vitesse
                    fig_speed_evolution = px.line(
                        points_df, 
                        x=range(len(points_df)), 
                        y="speed",
                        markers=True,
                        labels={"speed": "Vitesse (m/s)", "x": "Points"},
                        title=f"Évolution de la vitesse sur {selected_segment}"
                    )
                    
                    st.plotly_chart(fig_speed_evolution, use_container_width=True)
                    
                    # Statistiques du segment
                    st.subheader(f"Statistiques détaillées de {selected_segment}")
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("Nombre de points", len(points_df))
                    
                    with stat_col2:
                        st.metric("Distance", f"{selected_segment_data['distance_km']:.2f} km")
                    
                    with stat_col3:
                        st.metric("Vitesse moyenne", f"{selected_segment_data['avg_speed']:.2f} m/s")
                    
                    with stat_col4:
                        if "direct" not in selected_segment:
                            # Calculer l'efficacité du segment par rapport au direct
                            direct_segment = next((s for s in segment_stats if s["segment_id"] == "direct"), None)
                            if direct_segment:
                                efficacite = (direct_segment["distance_km"] / total_distance) * 100
                                st.metric("Efficacité", f"{efficacite:.1f}%", 
                                          delta=f"{100-efficacite:.1f}%" if efficacite < 100 else None,
                                          delta_color="inverse")
                else:
                    st.info(f"Aucune donnée disponible pour {selected_segment}")
            else:
                st.info("Aucun segment avec des données disponibles pour analyse détaillée")
        else:
            st.info("Aucun segment ne contient de données de points pour afficher des graphiques comparatifs")
elif page == "Analyse de vitesse":
    st.header("Analyse détaillée de la vitesse")
    
    if len(df) > 0:
        # Convertir timestamp en datetime si ce n'est pas déjà fait
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Graphique d'évolution de la vitesse
        st.subheader("Évolution de la vitesse")
        fig = px.line(
            df, 
            x=range(len(df)) if 'timestamp' not in df.columns else 'timestamp', 
            y='speed',
            labels={"speed": "Vitesse (m/s)", "x": "Points", "timestamp": "Temps"},
            title="Évolution de la vitesse au fil du temps"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des vitesses
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution des vitesses")
            fig_hist = px.histogram(
                df, 
                x='speed',
                nbins=20,
                labels={"speed": "Vitesse (m/s)", "count": "Nombre de points"},
                title="Distribution des vitesses"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Boxplot des vitesses")
            fig_box = px.box(
                df, 
                y='speed',
                labels={"speed": "Vitesse (m/s)"},
                title="Boxplot des vitesses"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistiques détaillées
        st.subheader("Statistiques de vitesse")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Vitesse moyenne", f"{df['speed'].mean():.2f} m/s")
        with stat_col2:
            st.metric("Vitesse maximale", f"{df['speed'].max():.2f} m/s")
        
        with stat_col3:
            st.metric("Vitesse minimale", f"{df['speed'].min():.2f} m/s")
        
        with stat_col4:
            st.metric("Écart type", f"{df['speed'].std():.2f} m/s")
        
        # Carte de chaleur pour vitesse vs. position
        st.subheader("Carte de chaleur des vitesses")
        
        # Créer une carte folium pour la heatmap
        m_heat = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=14)
        
        # Préparer les données pour la heatmap (lat, lon, speed comme poids)
        heat_data = [[row['latitude'], row['longitude'], row['speed']] for _, row in df.iterrows()]
        
        # Ajouter la heatmap
        HeatMap(
            data=heat_data,
            radius=15,
            gradient={"0.2": "blue", "0.4": "lime", "0.6": "yellow", "1.0": "red"},
            min_opacity=0.5,
            max_zoom=18
        ).add_to(m_heat)
        
        # Afficher la carte
        folium_static(m_heat, width=800, height=400)
    
    else:
        st.info("Aucune donnée disponible pour l'analyse de vitesse.")

elif page == "Clusters":
    st.header("Analyse des clusters")
    
    if len(df) < 10:
        st.warning("Pas assez de données pour l'analyse de clusters. Minimum 10 points requis.")
    else:
        # Vérifier si les modèles ML sont disponibles
        if not st.session_state.ml_models['is_trained']:
            st.warning("Modèles ML non entraînés. Veuillez les entraîner d'abord.")
            if st.button("Entraîner les modèles ML"):
                with st.spinner("Entraînement des modèles en cours..."):
                    success = train_ml_models()
                    if success:
                        st.success("Modèles ML entraînés avec succès!")
                        st.rerun()
                    else:
                        st.error("Échec de l'entraînement des modèles ML")
        else:
            # Paramètres du clustering
            st.subheader("Paramètres de clustering")
            
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                eps = st.slider("Distance maximale entre points (mètres)", 10, 200, 50)
            
            with param_col2:
                min_samples = st.slider("Nombre minimum de points par cluster", 3, 20, 5)
            
            # Bouton pour lancer l'analyse
            if st.button("Analyser les clusters"):
                with st.spinner("Analyse des clusters en cours..."):
                    # Préparation des coordonnées
                    coords = df[['latitude', 'longitude']].values
                    
                    # Conversion des degrés en radians
                    coords_rad = np.radians(coords)
                    
                    # Calcul du clustering
                    kms_per_radian = 6371.0088  # Rayon de la Terre en km
                    epsilon = eps / 1000 / kms_per_radian  # Conversion en radians
                    
                    db = DBSCAN(
                        eps=epsilon, 
                        min_samples=min_samples, 
                        algorithm='ball_tree',
                        metric='haversine'
                    ).fit(coords_rad)
                    
                    # Ajout des labels au DataFrame
                    df_clusters = df.copy()
                    df_clusters['cluster'] = db.labels_
                    
                    # Création de la carte avec clusters
                    m_clusters = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=14)
                    
                    # Couleurs pour les clusters
                    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                            'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
                    
                    # Créer un dictionnaire pour stocker les points de chaque cluster
                    cluster_points = {}
                    
                    for idx, row in df_clusters.iterrows():
                        cluster_id = int(row['cluster'])
                        
                        # Ajouter le point au dictionnaire pour traitement ultérieur
                        if cluster_id != -1:  # Ignorer les points de bruit pour les polygones
                            if cluster_id not in cluster_points:
                                cluster_points[cluster_id] = []
                            cluster_points[cluster_id].append([row['latitude'], row['longitude']])
                        
                        # Afficher les points sur la carte
                        if cluster_id == -1:
                            # Points de bruit
                            folium.CircleMarker(
                                [row['latitude'], row['longitude']],
                                radius=3,  # Plus petit pour les points de bruit
                                color='grey',
                                fill=True,
                                fill_opacity=0.5,
                                popup=f"Bruit: {row['speed']:.2f} m/s"
                            ).add_to(m_clusters)
                        else:
                            # Points de cluster
                            color_idx = cluster_id % len(colors)
                            folium.CircleMarker(
                                [row['latitude'], row['longitude']],
                                radius=5,
                                color=colors[color_idx],
                                fill=True,
                                fill_opacity=0.7,
                                popup=f"Cluster {cluster_id}: {row['speed']:.2f} m/s"
                            ).add_to(m_clusters)
                    
                    # Dessiner les polygones de cluster avec une couleur transparente
                    for cluster_id, points in cluster_points.items():
                        if len(points) >= 3:  # Besoin d'au moins 3 points pour un polygone
                            # Calculer l'enveloppe convexe
                            try:
                                points_array = np.array(points)
                                hull = ConvexHull(points_array)
                                
                                # Extraire les points de l'enveloppe
                                hull_points = [points_array[hull.vertices, :]]
                                
                                # Couleur du cluster
                                color_idx = cluster_id % len(colors)
                                
                                # Dessiner le polygone
                                folium.Polygon(
                                    locations=hull_points,
                                    color=colors[color_idx],
                                    fill=True,
                                    fill_color=colors[color_idx],
                                    fill_opacity=0.2,
                                    weight=2,
                                    popup=f"Zone Cluster {cluster_id}"
                                ).add_to(m_clusters)
                            except:
                                # Ignorer si le calcul de l'enveloppe échoue
                                pass
                    
                    # Ajouter les centres des clusters avec taille proportionnelle au nombre de points
                    for cluster_id, points in cluster_points.items():
                        # Nombre de points dans le cluster
                        num_points = len(points)
                        
                        # Calculer le centre du cluster
                        center_lat = sum(p[0] for p in points) / num_points
                        center_lon = sum(p[1] for p in points) / num_points
                        
                        # Couleur du cluster
                        color_idx = cluster_id % len(colors)
                        
                        # Calculer la taille de l'icône en fonction du nombre de points
                        # Limiter la taille pour les grands clusters
                        icon_size = min(30, 10 + (num_points / 10))
                        
                        # Ajouter un marqueur pour le centre
                        folium.CircleMarker(
                            [center_lat, center_lon],
                            radius=icon_size,
                            color=colors[color_idx],
                            fill=True,
                            fill_opacity=0.8,
                            weight=3,
                            popup=f"Centre Cluster {cluster_id}<br>Nombre de points: {num_points}"
                        ).add_to(m_clusters)
                        
                        # Ajouter un texte pour identifier le cluster
                        folium.map.Marker(
                            [center_lat, center_lon],
                            icon=folium.DivIcon(
                                icon_size=(20, 20),
                                icon_anchor=(10, 10),
                                html=f'<div style="font-size: 10pt; color: white; font-weight: bold; text-align: center;">{cluster_id}</div>'
                            )
                        ).add_to(m_clusters)
                    
                    # Afficher la carte avec clusters
                    st.subheader("Carte des clusters identifiés")
                    folium_static(m_clusters, width=800, height=500)
                    
                    # Statistiques des clusters
                    st.subheader("Statistiques des clusters")
                    
                    # Nombre de clusters
                    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                    n_noise = list(db.labels_).count(-1)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Nombre de clusters", n_clusters)
                    col2.metric("Points de bruit", n_noise)
# Distribution des points par cluster
                    if n_clusters > 0:
                        cluster_counts = df_clusters[df_clusters['cluster'] != -1]['cluster'].value_counts().sort_index()
                        fig = px.bar(
                            x=cluster_counts.index, 
                            y=cluster_counts.values,
                            labels={"x": "Cluster ID", "y": "Nombre de points"},
                            title="Distribution des points par cluster"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Vitesse moyenne par cluster
                    if n_clusters > 0:
                        cluster_speeds = df_clusters[df_clusters['cluster'] != -1].groupby('cluster')['speed'].mean()
                        fig = px.bar(
                            x=cluster_speeds.index, 
                            y=cluster_speeds.values,
                            labels={"x": "Cluster ID", "y": "Vitesse moyenne (m/s)"},
                            title="Vitesse moyenne par cluster"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # Tableau récapitulatif des clusters
                    if n_clusters > 0:
                        st.subheader("Récapitulatif des clusters")
                        
                        # Créer un dataframe récapitulatif
                        cluster_summary = []
                        
                        for cluster_id in range(n_clusters):
                            cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
                            
                            if len(cluster_data) > 0:
                                summary = {
                                    "Cluster ID": cluster_id,
                                    "Nombre de points": len(cluster_data),
                                    "Vitesse moyenne (m/s)": cluster_data['speed'].mean(),
                                    "Vitesse min (m/s)": cluster_data['speed'].min(),
                                    "Vitesse max (m/s)": cluster_data['speed'].max(),
                                    "Centre latitude": cluster_data['latitude'].mean(),
                                    "Centre longitude": cluster_data['longitude'].mean()
                                }
                                
                                if 'activity' in cluster_data.columns:
                                    top_activity = cluster_data['activity'].value_counts().idxmax()
                                    summary["Activité principale"] = top_activity
                                
                                cluster_summary.append(summary)
                        
                        summary_df = pd.DataFrame(cluster_summary)
                        st.dataframe(summary_df)

elif page == "Anomalies":
    st.header("Détection d'anomalies")
    
    if len(df) < 10:
        st.warning("Pas assez de données pour la détection d'anomalies. Minimum 10 points requis.")
    else:
        # Vérifier si les modèles ML sont disponibles
        if not st.session_state.ml_models['is_trained']:
            st.warning("Modèles ML non entraînés. Veuillez les entraîner d'abord.")
            if st.button("Entraîner les modèles ML"):
                with st.spinner("Entraînement des modèles en cours..."):
                    success = train_ml_models()
                    if success:
                        st.success("Modèles ML entraînés avec succès!")
                        st.rerun()
                    else:
                        st.error("Échec de l'entraînement des modèles ML")
        else:
            # Options pour la détection d'anomalies
            st.subheader("Paramètres de détection")
            
            anomaly_method = st.radio(
                "Méthode de détection d'anomalies",
                ["Basée sur un modèle", "Basée sur des règles"]
            )
            
            if anomaly_method == "Basée sur un modèle":
                # Option avec modèle d'apprentissage
                contamination = st.slider("Pourcentage d'anomalies attendu", 0.01, 0.2, 0.05, 0.01)
                features = st.multiselect(
                    "Caractéristiques à utiliser pour la détection",
                    ["speed", "latitude", "longitude"],
                    default=["speed"]
                )
                
                # Bouton pour exécuter la détection d'anomalies
                if st.button("Détecter les anomalies (modèle)"):
                    with st.spinner("Détection des anomalies en cours..."):
                        if not features:
                            st.error("Veuillez sélectionner au moins une caractéristique.")
                        else:
                            # Préparation des données
                            X = df[features].values
                            
                            # Entraînement du modèle Isolation Forest
                            model = IsolationForest(
                                contamination=contamination, 
                                random_state=42, 
                                n_estimators=100
                            )
                            
                            # Prédiction des anomalies
                            pred = model.fit_predict(X)
                            scores = model.decision_function(X)
                            
                            # Convertir les prédictions
                            # IsolationForest retourne 1 pour les points normaux et -1 pour les anomalies
                            df_anomalies = df.copy()
                            df_anomalies['anomaly'] = pred
                            df_anomalies['anomaly_score'] = scores
                            
                            # Afficher les résultats de la détection
                            display_anomaly_results(df_anomalies)
            
            else:  # Méthode basée sur des règles
                st.info("Cette méthode identifie les anomalies basées sur des règles simples définies par l'utilisateur.")
                
                # Définir les règles pour les anomalies
                speed_threshold_low = st.slider("Vitesse anormalement basse (m/s)", 0.0, 5.0, 0.1, 0.1)
                speed_threshold_high = st.slider("Vitesse anormalement élevée (m/s)", 5.0, 30.0, 15.0, 0.5)
                
                # Bouton pour exécuter la détection d'anomalies
                if st.button("Détecter les anomalies (règles)"):
                    with st.spinner("Détection des anomalies en cours..."):
                        # Créer une copie du DataFrame
                        df_anomalies = df.copy()
                        
                        # Définir les anomalies selon les règles
                        df_anomalies['anomaly'] = 1  # Par défaut, tous les points sont normaux
                        
                        # Marquer les anomalies (vitesse trop basse ou trop élevée)
                        df_anomalies.loc[df_anomalies['speed'] <= speed_threshold_low, 'anomaly'] = -1
                        df_anomalies.loc[df_anomalies['speed'] >= speed_threshold_high, 'anomaly'] = -1
                        
                        # Calcul d'un score d'anomalie simplifié
                        # Distance normalisée par rapport aux seuils
                        def calculate_score(speed):
                            if speed <= speed_threshold_low:
                                return -1 * (speed_threshold_low - speed) / speed_threshold_low
                            elif speed >= speed_threshold_high:
                                return -1 * (speed - speed_threshold_high) / speed_threshold_high
                            else:
                                # Distance relative au seuil le plus proche
                                dist_to_low = (speed - speed_threshold_low) / (speed_threshold_high - speed_threshold_low)
                                dist_to_high = (speed_threshold_high - speed) / (speed_threshold_high - speed_threshold_low)
                                return min(dist_to_low, dist_to_high)
                        
                        df_anomalies['anomaly_score'] = df_anomalies['speed'].apply(calculate_score)
                        
                        # Afficher les résultats de la détection
                        display_anomaly_results(df_anomalies)
            

elif page == "Statistiques":
    st.header("Statistiques générales")
    
    if len(df) == 0:
        st.info("Aucune donnée disponible pour les statistiques.")
    else:
        # Métriques de base
        col1, col2, col3 = st.columns(3)
        col1.metric("Total des points", len(df))
        col2.metric("Distance estimée", f"{calculate_distance(df):.2f} km")
        if 'timestamp' in df.columns:
            col3.metric("Durée", calculate_duration(df))
        
        # Graphiques supplémentaires
        col1, col2 = st.columns(2)
        
        with col1:
            # Nuage de points 2D (scatter plot)
            fig = px.scatter(df, x='longitude', y='latitude', color='speed',
                          color_continuous_scale='Viridis',
                          labels={"longitude": "Longitude", "latitude": "Latitude", "speed": "Vitesse (m/s)"},
                          title="Nuage de points géographiques")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Heatmap 2D des positions
            fig = px.density_heatmap(df, x='longitude', y='latitude',
                                   labels={"longitude": "Longitude", "latitude": "Latitude"},
                                   title="Distribution spatiale des points (heatmap)")
            st.plotly_chart(fig, use_container_width=True)
        
        if 'activity' in df.columns:
            st.subheader("Distribution par activité")
            fig = px.pie(df, names='activity', title="Répartition des activités")
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques avancées
        st.subheader("Statistiques avancées")
        
        col1, col2 = st.columns(2)
        
        # Créer des catégories de vitesse pour l'analyse
        speed_categories = {
            '0-1 m/s': 0,
            '1-2 m/s': 0,
            '2-5 m/s': 0,
            '5-10 m/s': 0,
            '10+ m/s': 0
        }
        
        # Compter les points dans chaque catégorie
        for speed in df['speed']:
            if speed < 1:
                speed_categories['0-1 m/s'] += 1
            elif speed < 2:
                speed_categories['1-2 m/s'] += 1
            elif speed < 5:
                speed_categories['2-5 m/s'] += 1
            elif speed < 10:
                speed_categories['5-10 m/s'] += 1
            else:
                speed_categories['10+ m/s'] += 1
        
        with col1:
            # Histogramme des vitesses par catégorie
            fig = px.bar(
                x=list(speed_categories.keys()),
                y=list(speed_categories.values()),
                labels={"x": "Catégorie de vitesse", "y": "Nombre de points"},
                title="Distribution des vitesses par catégorie"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'timestamp' in df.columns and df['timestamp'].dtype != 'object':
                # Extraire l'heure
                df_hour = df.copy()
                df_hour['hour'] = df_hour['timestamp'].dt.hour
                
                # Histogramme par heure
                fig = px.histogram(
                    df_hour,
                    x='hour',
                    labels={"hour": "Heure de la journée", "count": "Nombre de points"},
                    title="Distribution des points par heure"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Données temporelles non disponibles pour l'analyse horaire.")
        
        # Visualisation 3D de la densité des points
        st.subheader("Visualisation 3D de la densité")
        
        # Créer un histogramme 2D pour la visualisation 3D
        hist, x_edges, y_edges = np.histogram2d(
            df['longitude'], 
            df['latitude'], 
            bins=[20, 20]
        )
        
        # Coordonnées X et Y pour le graphique 3D
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
        
        # Créer la visualisation 3D avec Plotly
        fig = go.Figure(data=[go.Surface(
            z=hist.T,
            x=x_mesh,
            y=y_mesh,
            colorscale='Viridis',
            opacity=0.8
        )])
        
        fig.update_layout(
            title="Distribution 3D des points",
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Nombre de points'
            ),
            width=800,
            height=600,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les données brutes
        if st.checkbox("Afficher les données brutes"):
            st.subheader("Données brutes")
            st.dataframe(df)

elif page == "Configuration":
    st.header("Configuration du système")
    
    # Onglets pour différentes options de configuration
    config_tabs = st.tabs(["Ajout manuel", "Import CSV", "Export", "Modèles ML"])
    
    # Onglet Ajout manuel
    with config_tabs[0]:
        st.subheader("Ajouter manuellement un point")
        
        # Formulaire pour ajouter un point
        with st.form("add_point_form"):
            point_col1, point_col2, point_col3 = st.columns(3)
            
            with point_col1:
                latitude = st.number_input("Latitude", value=49.2484420, format="%.7f")
            
            with point_col2:
                longitude = st.number_input("Longitude", value=4.0415017, format="%.7f")
            
            with point_col3:
                speed = st.number_input("Vitesse (m/s)", value=1.0, min_value=0.0, format="%.2f")
            
            submit_button = st.form_submit_button("Ajouter ce point")
            
        if submit_button:
            # Ajouter le point à la base de données
            result = push_data(latitude, longitude, speed)
            
            if result["status"] == "success":
                st.success(f"Point ajouté avec succès! Vitesse: {speed} m/s")
                
                # Afficher la carte avec le nouveau point
                m = folium.Map(location=[latitude, longitude], zoom_start=16)
                
                # Ajouter le marqueur
                folium.Marker(
                    [latitude, longitude],
                    popup=f"Nouveau point: {speed} m/s",
                    icon=folium.Icon(color="green")
                ).add_to(m)
                
                folium_static(m, width=700, height=300)
                
                # Proposer d'actualiser les données
                if st.button("Actualiser les données"):
                    st.rerun()
            else:
                st.error(f"Erreur lors de l'ajout du point: {result.get('message', 'Erreur inconnue')}")
    
    # Onglet Import CSV
    with config_tabs[1]:
        st.subheader("Importer des données CSV")
        
        # Uploader un fichier CSV
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        
        if uploaded_file is not None:
            # Analyser les données
            try:
                csv_data = pd.read_csv(uploaded_file)
                
                # Vérifier si les colonnes requises sont présentes
                required_cols = ['latitude', 'longitude', 'speed']
                missing_cols = [col for col in required_cols if col not in csv_data.columns]
                
                if missing_cols:
                    st.error(f"Colonnes manquantes dans le CSV: {', '.join(missing_cols)}")
                    st.info("Le CSV doit contenir au minimum les colonnes: latitude, longitude et speed")
                else:
                    # Afficher un aperçu
                    st.subheader("Aperçu des données")
                    st.dataframe(csv_data.head())
                    
                    # Afficher les statistiques
                    st.subheader("Statistiques")
                    st.write(f"Nombre de points: {len(csv_data)}")
                    st.write(f"Vitesse moyenne: {csv_data['speed'].mean():.2f} m/s")
                    
                    # Option pour importer
                    if st.button("Importer ces données"):
                        with st.spinner("Importation en cours..."):
                            # Créer une session de base de données
                            session = Session()
                            
                            # Compter les lignes importées
                            success_count = 0
                            errors = []
                            
                            for _, row in csv_data.iterrows():
                                try:
                                    # Extraire les données
                                    lat = float(row['latitude'])
                                    lon = float(row['longitude'])
                                    spd = float(row['speed'])
                                    
                                    # Créer un nouvel enregistrement
                                    new_data = SensorData(
                                        latitude=lat,
                                        longitude=lon,
                                        speed=spd
                                    )
                                    
                                    # Ajouter à la session
                                    session.add(new_data)
                                    success_count += 1
                                except Exception as e:
                                    errors.append(str(e))
                            
                            # Valider la transaction
                            if success_count > 0:
                                session.commit()
                            
                            session.close()
                            
                            # Afficher les résultats
                            if success_count > 0:
                                st.success(f"{success_count} points importés avec succès!")
                                
                                if errors:
                                    st.warning(f"{len(errors)} erreurs rencontrées.")
                                    if st.checkbox("Afficher les erreurs"):
                                        for i, err in enumerate(errors[:10]):  # Limiter à 10 erreurs
                                            st.text(f"Erreur {i+1}: {err}")
                                
                                # Proposer d'actualiser les données
                                if st.button("Actualiser les données"):
                                    st.rerun()
                            else:
                                st.error("Aucun point n'a pu être importé.")
                                if errors:
                                    st.error(f"Erreurs: {errors[0]}")
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse du CSV: {str(e)}")
    
    # Onglet Export
    with config_tabs[2]:
        st.subheader("Exporter les données")
        
        export_format = st.selectbox("Format d'export", ["CSV", "JSON"])
        
        if st.button("Exporter les données"):
            # Récupérer toutes les données
            data = get_all_points()
            
            if data["status"] == "success" and data["points"]:
                points_df = pd.DataFrame(data["points"])
                
                # Créer le fichier d'export
                if export_format == "CSV":
                    csv_export = points_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger le CSV",
                        data=csv_export,
                        file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:  # JSON
                    json_export = points_df.to_json(orient="records")
                    st.download_button(
                        label="Télécharger le JSON",
                        data=json_export,
                        file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                st.success(f"{len(points_df)} points exportés avec succès!")
            else:
                st.error("Aucune donnée disponible pour l'export.")
    
    # Onglet Modèles ML
    with config_tabs[3]:
        st.subheader("Gestion des modèles ML")
        
        # Vérifier l'état des modèles
        is_trained = st.session_state.ml_models['is_trained']
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            status_text = "✅ Entraînés" if is_trained else "❌ Non entraînés"
            st.info(f"Statut des modèles ML: {status_text}")
        
        with status_col2:
            session = Session()
            point_count = session.query(SensorData).count()
            session.close()
            
            st.info(f"Nombre de points disponibles: {point_count}")
            if point_count < 10:
                st.warning("Minimum 10 points requis pour l'entraînement")
        
        # Options pour gérer les modèles
        if is_trained:
            if st.button("Réentraîner les modèles"):
                with st.spinner("Réentraînement en cours..."):
                    success = train_ml_models()
                    if success:
                        st.success("Modèles ML réentraînés avec succès!")
                    else:
                        st.error("Échec du réentraînement des modèles ML")
            
            # Option pour exporter les modèles
            if st.button("Exporter les modèles"):
                try:
                    model_dir = "ml_models"
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Sauvegarder les modèles
                    if st.session_state.ml_models['isolation_forest'] is not None:
                        joblib.dump(
                            st.session_state.ml_models['isolation_forest'], 
                            os.path.join(model_dir, "iso_forest.pkl")
                        )
                    
                    if st.session_state.ml_models['dbscan'] is not None:
                        joblib.dump(
                            st.session_state.ml_models['dbscan'], 
                            os.path.join(model_dir, "dbscan.pkl")
                        )
                    
                    st.success("Modèles ML exportés avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de l'exportation des modèles: {str(e)}")
        else:
            if st.button("Entraîner les modèles"):
                with st.spinner("Entraînement en cours..."):
                    if point_count >= 10:
                        success = train_ml_models()
                        if success:
                            st.success("Modèles ML entraînés avec succès!")
                            # Recommander de rafraîchir la page
                            st.info("Veuillez actualiser la page pour utiliser les modèles")
                            if st.button("Actualiser la page"):
                                st.rerun()
                        else:
                            st.error("Échec de l'entraînement des modèles ML")
                    else:
                        st.error("Pas assez de données pour l'entraînement (minimum 10 points)")

# Démarrer le serveur Flask à la fin de votre fichier
if __name__ == "__main__":
    # Démarrer Flask dans un thread séparé
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.daemon = True  # Le thread se terminera quand Streamlit se termine
    flask_thread.start()
    
    # Informer l'utilisateur que l'API est démarrée
    print(f"🚀 API Flask démarrée sur le port 5001 pour recevoir les données de l'app Swift")
    
    # Le reste du code de démarrage de Streamlit
    if not st.session_state.ml_models['is_trained']:
        load_ml_models()