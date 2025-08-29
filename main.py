from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI(title="Re:Mind Backend")

# CORS: allow your Vercel/local dev to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    user_id: str
    message: str
    now_iso: str
    tz: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(p: ChatIn):
    """
    Minimal, safe fallback: always return a reply.
    (Your real Groq + Supabase logic can be added later.)
    """
    try:
        user_text = (p.message or "").strip()
        if not user_text:
            return {"reply": "I didn’t receive any text. Try: add \"Buy milk\" tomorrow 8am."}

        # 🔒 Keep it simple for now so we never 500:
        return {"reply": f"Re:Mind noted: {user_text}"}

    except Exception as e:
        logging.exception("chat endpoint failed")
        # Still return 200 with a helpful message so the UI never breaks
        return {"reply": "Something went wrong on the server, but I’m still here. Try again in a moment."}
