from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

app = FastAPI(title="Geo-Intellix MVP")

# 1. Enable CORS (So your React frontend can talk to this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for MVP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup a local folder to simulate S3 storage
UPLOAD_DIR = "temp_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Geo-Intellix System Online"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Step 1 MVP Endpoint:
    Accepts a file, saves it locally, and returns a Mock JSON.
    """
    # Generate a unique Job ID
    job_id = str(uuid.uuid4())
    
    # Define where to save the raw file
    file_location = f"{UPLOAD_DIR}/{job_id}_{file.filename}"
    
    # Save the file (Simulating upload to S3)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"Received file: {file.filename} -> Saved to {file_location}")

    # RETURN THE MOCK RESPONSE (The "Fake" Model)
    # This proves to the frontend team that the pipeline "works"
    return {
        "job_id": job_id,
        "status": "success",
        "filename": file.filename,
        "text_answer": "MVP TEST: The image shows dense urban construction near the river bank. Detected 12 new structures.",
        "map_url": "https://via.placeholder.com/600x400.png?text=Annotated+Map+Placeholder",
        "summary_metrics": {
            "area_ha": 12.5,
            "num_buildings": 24
        }
    }