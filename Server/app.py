from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import os
import json

app = FastAPI()

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/upload")
def get_upload_info():
    """
    Endpoint to confirm upload availability.
    """
    return {"message": "Upload endpoint is live. Use POST to upload a file."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file to the server.
    """
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return JSONResponse(content={"message": f"File {file.filename} uploaded successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/uploads/{file_name}")
def get_uploaded_file(file_name: str):
    """
    Endpoint to serve an uploaded file.
    """
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)

@app.post("/result")
async def receive_result(data: dict):
    """
    Endpoint to receive prediction results from Colab.
    """
    try:
        predictions = data.get("predictions", [])
        if not predictions:
            return JSONResponse(content={"error": "No predictions received"}, status_code=400)

        # Save predictions to a file
        result_file = os.path.join(UPLOAD_FOLDER, "predictions.json")
        with open(result_file, "w") as f:
            json.dump(predictions, f)

        return JSONResponse(content={"message": "Results received successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
