from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
import shutil

# Import the Vision Logic
from verifyre_auditor import get_fraud_probability
# try:
#     from verifyre_auditor import get_fraud_probability
# except ImportError:
#     # Fail-safe / Simulation if model is missing during raw API test
#     print("WARNING: Could not import vision engine. Running in MOCK mode.")
#     get_fraud_probability = None

app = FastAPI(title="Verifyre Satellite Audit API", version="1.0")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow localhost frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES ---
# Create directory for heatmaps if it doesn't exist
os.makedirs("static/heatmaps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- MODELS ---
class AuditRequest(BaseModel):
    sector_id: str
    claim: str

# --- DATA PREP FOR DEMO ---
# Ensure we have the hardcoded demo images available in static/heatmaps
# In a real app we'd generate these on startup, here we assume they exist or we copy them from data/
# For this script, to be safe, I'll generate dummy files if missing so it doesn't 404.

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "system": "Verifyre Satellite Uplink v2.0"}

@app.post("/audit")
def trigger_audit(request: AuditRequest):
    """
    Live Audit Endpoint.
    Connects to the Computer Vision Audit Engine.
    """
    if not get_fraud_probability:
        return {"error": "Vision Engine Offline"}
    
    # Run the real audit logic (which picks a random satellite image to simulate the sector)
    result = get_fraud_probability(request.sector_id, request.claim)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@app.post("/demo/{scenario}")
def demo_scenario(scenario: str):
    """
    Live Demo Scenarios.
    Runs the REAL computer vision model on the specific 'Drax' or 'Clean' images.
    """
    # 1. FRAUD SCENARIO (Drax Biomass)
    if scenario == "fraud":
        # Target: The controversial photo
        target_image = "static/sector-4_view.jpg" 
        
        # If image missing, ensure we don't crash (though it should exist)
        if not os.path.exists(target_image):
            return {"error": f"Demo asset missing: {target_image}"}

        # CALL THE MODEL LIVE
        return get_fraud_probability(
            sector_id="SEC-999-DEMO", 
            claim="Forest", 
            image_path=target_image
        )
    
    # 2. CLEAN SCENARIO (GreenGuard)
    elif scenario == "clean":
        target_image = "static/sector-1_view.jpg"
        
        if not os.path.exists(target_image):
            return {"error": f"Demo asset missing: {target_image}"}

        # CALL THE MODEL LIVE
        return get_fraud_probability(
            sector_id="SEC-001-DEMO", 
            claim="Forest", 
            image_path=target_image
        )
        
    else:
        raise HTTPException(status_code=404, detail="Scenario unknown. Use 'fraud' or 'clean'.")

if __name__ == "__main__":
    import uvicorn
    # Updated Port to 8001 to avoid conflict with Reflex (8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)
