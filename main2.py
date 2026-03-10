from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

# Import your helper scripts
from resume_parser import extract_text_from_pdf
from gemini_service import GeminiService

app = FastAPI()
gemini = GeminiService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# IN-MEMORY DATABASE: This replaces MongoDB for your demo
# It clears whenever you restart the server.
active_interviews = {}

@app.get("/")
def home():
    return {"message": "AI Interview Backend (Local Memory Mode) Running"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resume_text = extract_text_from_pdf(file_path)
    role_type = gemini.determine_role_type(resume_text)
    
    # Store interview state in our global dictionary
    active_interviews[session_id] = {
        "current_step": 1,
        "resume_text": resume_text,
        "role_type": role_type,
        "responses": []
    }

    first_question = gemini.get_interview_question(1, resume_text, role_type)

    return {
        "session_id": session_id,
        "role_detected": role_type,
        "question": first_question,
        "step": 1
    }

@app.post("/submit-answer")
async def submit_answer(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    if session_id not in active_interviews:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_interviews[session_id]
    current_step = session["current_step"]
    
    # Save audio
    audio_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_step_{current_step}.wav")
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # 1. Transcribe & Evaluate
    transcription = gemini.transcribe_audio(audio_path)
    current_q = gemini.get_interview_question(current_step, session["resume_text"], session["role_type"])
    evaluation = gemini.evaluate_response(current_q, transcription)

    # 2. Update Session Data
    session["responses"].append({
        "question": current_q,
        "answer": transcription,
        "score": evaluation.get("score")
    })

    # 3. Increment Step
    next_step = current_step + 1
    session["current_step"] = next_step

    if next_step > 5:
        # Instead of saving to Mongo, we just return the final report
        report = {
            "message": "Interview Completed!",
            "is_complete": true,
            "final_score": evaluation.get("score"),
            "feedback": evaluation.get("feedback")
        }
        return report

    # 4. Generate next question + Voice
    next_question = gemini.get_interview_question(next_step, session["resume_text"], session["role_type"])
    audio_url = gemini.text_to_speech(next_question, UPLOAD_FOLDER)

    return {
        "session_id": session_id,
        "next_step": next_step,
        "next_question": next_question,
        "audio_url": audio_url,
        "is_complete": False
    }