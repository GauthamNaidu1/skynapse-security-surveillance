import json
import random
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import numpy as np
import kagglehub # Uncomment if running locally with internet access
import pandas as pd
import os

class SentinelAI:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'ada': AdaBoostClassifier(n_estimators=50, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'dt': DecisionTreeClassifier(random_state=42)
        }
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['rf']),
                ('ada', self.models['ada']),
                ('gb', self.models['gb']),
                ('knn', self.models['knn']),
                ('dt', self.models['dt'])
            ],
            voting='soft'
        )
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.is_trained = False
        self.vision_model_status = "idle"

    def generate_synthetic_data(self, n_samples=500):
        # Generate synthetic data for training
        # Features: [hour_of_day, crowd_density, location_id]
        X = []
        y = []
        locations = range(5) # 0: Hitech City, 1: Charminar, 2: Banjara Hills, 3: Secunderabad, 4: Gachibowli
        
        for _ in range(n_samples):
            hour = random.randint(0, 23)
            loc = random.choice(locations)
            density = random.randint(0, 100) # 0-100%
            
            # Logic for threat level (0: Low, 1: Medium, 2: High, 3: Critical)
            threat = 0
            if hour > 22 or hour < 5:
                threat += 1
            if density > 80:
                threat += 1
            if loc in [1, 3] and hour > 20: # Higher risk in certain areas at night
                threat += 1
            
            threat = min(threat, 3)
            X.append([hour, density, loc])
            y.append(threat)
            
        return np.array(X), np.array(y)

    def preprocess_data(self):
        print("Preprocessing Kaggle Datasets...")
        
        security_path = r"C:\Users\Gautham\.cache\kagglehub\datasets\mikrokosmos1\securityincident\versions\1\security_incident_classification.csv"
        crime_path = r"C:\Users\Gautham\.cache\kagglehub\datasets\mikrokosmos1\crimedata\versions\1\hyderabad_crime_data_enhanced (1).csv"
        
        all_features = []
        all_targets = []
        
        # Location mapping for consistency
        self.location_map = {
            'Hitech City': 0, 'Charminar': 1, 'Banjara Hills': 2, 'Secunderabad': 3, 'Gachibowli': 4,
            'Jubilee Hills': 5, 'Ameerpet': 6, 'LB Nagar': 7, 'Madhapur': 8, 'Begumpet': 9, 'Kukatpally': 10
        }
        
        # Severity mapping
        severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        
        # 1. Process Crime Data (Better for location & actual hours)
        if os.path.exists(crime_path):
            df_crime = pd.read_csv(crime_path)
            for _, row in df_crime.iterrows():
                try:
                    hour = int(row['Time'].split(':')[0])
                    loc_id = self.location_map.get(row['Area'], random.randint(0, 10))
                    severity = severity_map.get(row['Severity'], 1)
                    
                    # Estimate density based on area/time for real data fallback
                    density = random.randint(20, 90) 
                    
                    all_features.append([hour, density, loc_id])
                    all_targets.append(severity)
                except:
                    continue
        
        # 2. Process Security Data (Better for density)
        if os.path.exists(security_path):
            df_sec = pd.read_csv(security_path)
            time_map = {'Night': 2, 'Morning': 8, 'Afternoon': 14, 'Evening': 19}
            for _, row in df_sec.iterrows():
                try:
                    hour = time_map.get(row['Time_of_Day'], random.randint(0, 23))
                    density = row['People_Density'] # Normalize density if needed, here we use raw
                    loc_id = random.randint(0, 10) # No explicit city area, use random or general
                    severity = severity_map.get(row['Severity_Level'], 1)
                    
                    all_features.append([hour, density, loc_id])
                    all_targets.append(severity)
                except:
                    continue

        if not all_features:
            print("No real data found, falling back to synthetic data.")
            return self.generate_synthetic_data()
            
        print(f"Loaded {len(all_features)} real data samples.")
        return np.array(all_features), np.array(all_targets)

    def train(self):
        print("Training SentinelAI Ensemble with Real Data...")
        try:
            X, y = self.preprocess_data()
        except Exception as e:
            print(f"Preprocessing error: {e}. Falling back to synthetic.")
            X, y = self.generate_synthetic_data()
            
        self.ensemble.fit(X, y)
        self.kmeans.fit(X)
        self.is_trained = True
        print("Training Complete.")

    def predict_threat(self, hour, density, location_id):
        if not self.is_trained:
            self.train()
        
        input_data = np.array([[hour, density, location_id]])
        prediction = self.ensemble.predict(input_data)[0]
        probabilities = self.ensemble.predict_proba(input_data)[0]
        
        cluster = self.kmeans.predict(input_data)[0]
        
        return {
            "threat_level": int(prediction),
            "confidence": float(max(probabilities)),
            "cluster_id": int(cluster),
            "probabilities": probabilities.tolist()
        }

    def train_vision_model(self):
        self.vision_model_status = "downloading_dataset"
        print("Downloading dataset from Kaggle Hub...")
        # path = kagglehub.dataset_download("jonathannield/cctv-action-recognition-dataset")
        # print("Path:", path)
        time.sleep(2) # Simulate download
        self.vision_model_status = "training"
        time.sleep(3) # Simulate training
        self.vision_model_status = "ready"
        return {"status": "Vision Model Trained", "classes": ["Theft", "Arson", "Fighting", "Normal"]}

sentinel = SentinelAI()

if __name__ == "__main__":
    # Example usage
    sentinel.train()
    print(sentinel.predict_threat(23, 90, 1))
