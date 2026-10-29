import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Model artifact path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crime_pipeline.joblib")
LOCATIONS = [
    "Hitech City", "Charminar", "Banjara Hills", "Secunderabad", "Gachibowli",
    "Jubilee Hills", "Ameerpet", "LB Nagar", "Madhapur", "Begumpet",
    "Kukatpally", "Dilsukhnagar", "Mehdipatnam", "Uppal", "Miyapur",
    "Kondapur", "Tarnaka", "Abids", "Nampally", "Malakpet"
]

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            return None
        import joblib
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def format_features(hour: float, density: float, location_id: int, area_name: str = ""):
    """
    Format request inputs into a DataFrame matching the ColumnTransformer schema.
    """
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    now = datetime.now()
    day_of_week = int(now.weekday())
    is_weekend = int(day_of_week >= 5)

    area = area_name.strip()
    if not area:
        area = LOCATIONS[location_id] if 0 <= location_id < len(LOCATIONS) else "Hitech City"

    data = {
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'trailing_30d_density': [float(density)],
        'Area': [area]
    }

    # Include area one-hot columns for pipeline compatibility
    for loc in LOCATIONS:
        data[f"area_{loc}"] = [1 if loc == area else 0]

    return pd.DataFrame(data)

def predict_threat(hour: float, density: float, location_id: int, area_name: str = ""):
    pipeline = get_pipeline()
    if pipeline is None:
        return {
            "error": "Model artifact not found. Run python/evaluate_models.py to build models/crime_pipeline.joblib",
            "source": "heuristic fallback",
            "is_fallback": True
        }

    input_df = format_features(hour, density, location_id, area_name)
    
    # Run prediction through scikit-learn pipeline
    probs = pipeline.predict_proba(input_df)[0]
    prediction = int(pipeline.predict(input_df)[0])

    # Ensure 3-class probabilities (Low, Medium, High)
    prob_list = [float(p) for p in probs]
    if len(prob_list) == 2:
        prob_list.append(0.0)

    confidence = float(np.max(probs))

    return {
        "threat_level": prediction,
        "confidence": confidence,
        "probabilities": prob_list,
        "classes": ["Low", "Medium", "High"],
        "source": "ML Model (crime_pipeline.joblib)",
        "is_fallback": False,
        "cluster_id": prediction
    }

# -----------------------------------------------------------------------------
# FASTAPI IMPLEMENTATION
# -----------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Skynapse Crime Predictor API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class PredictionInput(BaseModel):
        hour: float
        density: float
        location_id: int = 0
        area_name: str = ""

    @app.get("/health")
    def health():
        pipeline_loaded = os.path.exists(MODEL_PATH)
        return {
            "status": "healthy",
            "model_loaded": pipeline_loaded,
            "model_path": MODEL_PATH
        }

    @app.post("/predict")
    def predict(req: PredictionInput):
        res = predict_threat(req.hour, req.density, req.location_id, req.area_name)
        if "error" in res and res.get("is_fallback"):
            raise HTTPException(status_code=503, detail=res["error"])
        return res

    def start_server():
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        print(f"[Starting FastAPI Service] http://{host}:{port} ...")
        uvicorn.run(app, host=host, port=port)

except ImportError:
    # -------------------------------------------------------------------------
    # FALLBACK STANDARD-LIBRARY HTTP SERVER (If fastapi/uvicorn not yet installed)
    # -------------------------------------------------------------------------
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class SimpleRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                resp = {
                    "status": "healthy (http.server fallback)",
                    "model_loaded": os.path.exists(MODEL_PATH),
                    "note": "Install fastapi and uvicorn for full ASGI FastAPI server: pip install fastapi uvicorn"
                }
                self.wfile.write(json.dumps(resp).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/predict":
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                try:
                    payload = json.loads(post_body.decode('utf-8'))
                    hour = float(payload.get('hour', 12))
                    density = float(payload.get('density', 50))
                    location_id = int(payload.get('location_id', 0))
                    area_name = str(payload.get('area_name', ''))

                    res = predict_threat(hour, density, location_id, area_name)
                    if "error" in res and res.get("is_fallback"):
                        self.send_response(503)
                    else:
                        self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode('utf-8'))
                except Exception as e:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

    def start_server():
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        server_address = (host, port)
        print(f"[Starting Standalone Predictor Service] Running on http://{host}:{port} ...")
        print("[Tip] To run with ASGI FastAPI: pip install fastapi uvicorn")
        httpd = HTTPServer(server_address, SimpleRequestHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down service.")

if __name__ == "__main__":
    start_server()
