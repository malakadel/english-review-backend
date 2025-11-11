from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import whisper
import language_tool_python
import os

app = FastAPI()

# Allow browser extension requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight Whisper model
model = whisper.load_model("tiny")
tool = language_tool_python.LanguageTool('en-US')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze_audio/")
async def analyze_audio(file: UploadFile = File(...)):
    # Save the uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    # Transcribe using Whisper
    result = model.transcribe(temp_audio_path)
    text = result['text'].strip()

    # Simple grammar check
    matches = tool.check(text)
    grammar_issues = [m.message for m in matches]

    os.remove(temp_audio_path)
    return {
        "transcription": text,
        "grammar_issues": grammar_issues,
        "pronunciation_score": 9.0 if len(text.split()) > 4 else 5.0
    }
