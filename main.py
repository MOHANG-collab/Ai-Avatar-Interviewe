from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

# Import your helper scripts
from resume_parser import extract_text_from_pdf
from gemini_service import GeminiService

app = FastAPI()
gemini = GeminiService()

# --- CORS Setup ---
# This allows your HTML file to talk to the Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Folder Setup ---
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mount the static folder so the browser can find the generated audio files
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# In-memory storage (Replaces MongoDB for your demo)
active_interviews = {}

@app.get("/")
def home():
    return {"status": "AI Interview Backend is Running Locally"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    # Generate a unique ID for this interview session
    session_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")

    # Save the uploaded PDF
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text and analyze the role
    resume_text = extract_text_from_pdf(file_path)
    role_type = gemini.determine_role_type(resume_text)
    
    # Initialize the interview session in memory
    active_interviews[session_id] = {
        "current_step": 1,
        "resume_text": resume_text,
        "role_type": role_type,
        "responses": []
    }

    # Get the first intro question
    first_question = gemini.get_interview_question(1, resume_text, role_type)
    
    return {
        "session_id": session_id,
        "question": first_question,
        "role_detected": role_type
    }

@app.post("/submit-answer")
async def submit_answer(
    session_id: str = Form(...), 
    audio_file: UploadFile = File(...)
):
    # Check if the session exists in our memory
    if session_id not in active_interviews:
        raise HTTPException(status_code=404, detail="Interview session expired or not found")

    session = active_interviews[session_id]
    current_step = session["current_step"]

    # Save the user's audio answer
    audio_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_ans_{current_step}.wav")
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # 1. Transcribe the user's speech using Gemini
    transcription = gemini.transcribe_audio(audio_path)
    
    # 2. Get the question that was just asked to evaluate the answer
    current_q = gemini.get_interview_question(current_step, session["resume_text"], session["role_type"])
    
    # 3. Evaluate the answer
    evaluation = gemini.evaluate_response(current_q, transcription)

    # 4. Save response data
    session["responses"].append({
        "step": current_step,
        "question": current_q,
        "answer": transcription,
        "score": evaluation.get("score")
    })

    # 5. Move to next question
    next_step = current_step + 1
    session["current_step"] = next_step

    # If 5 questions are finished, end the interview
    if next_step > 5:
        final_results = {
            "is_complete": True,
            "evaluation": evaluation,
            "all_responses": session["responses"]
        }
        # Clean up memory
        del active_interviews[session_id]
        return final_results

    # 6. Generate next question and convert to Voice (MP3)
    next_question = gemini.get_interview_question(next_step, session["resume_text"], session["role_type"])
    audio_url = gemini.text_to_speech(next_question, UPLOAD_FOLDER)

    return {
        "next_question": next_question,
        "audio_url": audio_url,
        "current_step": next_step,
        "is_complete": False
    }