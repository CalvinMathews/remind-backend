import os, json, re, uuid, datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from supabase import create_client, Client
from groq import Groq
import dateparser

# --- Environment ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# OPTIONAL: default sender email to show in confirmations (purely UI)
DEFAULT_SENDER = os.environ.get("REMIND_SENDER_EMAIL", "[email protected]")

# --- Init clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
groq = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Re:Mind Backend", version="1.0")

# === Pydantic models ===
class ChatRequest(BaseModel):
    user_id: str                  # Supabase user id (from frontend after login)
    message: str
    now_iso: Optional[str] = None # current time from client (optional)
    tz: Optional[str] = None      # e.g., "Asia/Kolkata" (optional)

# helper: parse date/time like "tomorrow 2pm"
def parse_datetime(text: str, now: Optional[str] = None, tz: Optional[str] = None) -> Optional[str]:
    settings = {"PREFER_DATES_FROM": "future"}
    if tz: settings["TIMEZONE"] = tz
    if now:
        dt = dateparser.parse(text, settings=settings, languages=["en"], RELATIVE_BASE=datetime.datetime.fromisoformat(now.replace("Z","+00:00")))
    else:
        dt = dateparser.parse(text, settings=settings, languages=["en"])
    if not dt: return None
    # normalize to ISO UTC
    if not dt.tzinfo:
        dt = dt.astimezone(datetime.timezone.utc) if hasattr(dt, "astimezone") else dt
    try:
        dt_utc = dt.astimezone(datetime.timezone.utc)
    except Exception:
        dt_utc = dt
    return dt_utc.replace(microsecond=0).isoformat().replace("+00:00","Z")

# --- DB helpers ---
def add_task(user_id: str, title: str, description: Optional[str], due_at_iso: Optional[str], source_text: str) -> Dict[str, Any]:
    payload = {
        "user_id": user_id,
        "title": title.strip(),
        "description": (description or "").strip() or None,
        "due_at": due_at_iso,
        "source_text": source_text
    }
    res = supabase.table("tasks").insert(payload).execute()
    task = res.data[0]
    # create reminder if due_at provided
    if due_at_iso:
        supabase.table("reminders").insert({
            "task_id": task["id"], "remind_at": due_at_iso, "method": "email"
        }).execute()
    return task

def update_task(user_id: str, title_contains: str, new_title: Optional[str]=None, new_due_at_iso: Optional[str]=None, new_status: Optional[str]=None) -> Dict[str, Any]:
    # find latest matching OPEN task by title substring
    q = supabase.table("tasks").select("*").eq("user_id", user_id).eq("status", "open").order("created_at", desc=True).execute()
    matches = [t for t in q.data if title_contains.lower() in t["title"].lower()]
    if not matches: raise HTTPException(404, "Task not found")
    task = matches[0]
    update = {}
    if new_title: update["title"] = new_title
    if new_status: update["status"] = new_status
    if new_due_at_iso is not None: update["due_at"] = new_due_at_iso
    if update:
        update["updated_at"] = datetime.datetime.utcnow().isoformat()+"Z"
        supabase.table("tasks").update(update).eq("id", task["id"]).execute()
    # refresh
    task = supabase.table("tasks").select("*").eq("id", task["id"]).single().execute().data
    # manage reminder row
    if new_due_at_iso is not None:
        # upsert reminder for this task: delete old pending, insert new
        supabase.table("reminders").delete().eq("task_id", task["id"]).eq("sent", False).execute()
        if new_due_at_iso:
            supabase.table("reminders").insert({
                "task_id": task["id"], "remind_at": new_due_at_iso, "method": "email"
            }).execute()
    return task

def delete_task(user_id: str, title_contains: str) -> int:
    q = supabase.table("tasks").select("id,title").eq("user_id", user_id).order("created_at", desc=True).execute()
    matches = [t for t in q.data if title_contains.lower() in t["title"].lower()]
    if not matches: raise HTTPException(404, "Task not found")
    tid = matches[0]["id"]
    supabase.table("tasks").delete().eq("id", tid).execute()
    return 1

def list_tasks(user_id: str, when: Optional[str]=None) -> List[Dict[str, Any]]:
    query = supabase.table("tasks").select("*").eq("user_id", user_id).order("due_at", desc=False)
    if when == "today":
        today = datetime.datetime.utcnow().date()
        start = datetime.datetime.combine(today, datetime.time.min, tzinfo=datetime.timezone.utc).isoformat().replace("+00:00","Z")
        end   = datetime.datetime.combine(today, datetime.time.max, tzinfo=datetime.timezone.utc).isoformat().replace("+00:00","Z")
        query = query.gte("due_at", start).lte("due_at", end)
    res = query.execute()
    return res.data or []

def store_memory(user_id: str, label: Optional[str], content: str) -> Dict[str, Any]:
    res = supabase.table("memories").insert({"user_id": user_id, "label": label, "content": content}).execute()
    return res.data[0]

def recall_memory(user_id: str, question: Optional[str], label_contains: Optional[str]) -> Optional[Dict[str, Any]]:
    q = supabase.table("memories").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
    if label_contains:
        for m in q:
            if m["label"] and label_contains.lower() in m["label"].lower():
                return m
    if question:
        # naive contains search in content
        for m in q:
            if question.lower() in (m["content"] or "").lower():
                return m
    return q[0] if q else None

# --- LLM intent extraction (JSON schema) ---
SYSTEM = """You are Re:Mind's planner. Convert user messages into a single JSON command.
Schema:
{ "action": "add_task|update_task|delete_task|list_tasks|store_memory|recall_memory|help",
  "title": "...", "description": "...", "when": "...", "status": "open|done|cancelled",
  "label": "...", "question": "...", "match": "substring to find existing task"
}
Rules:
- If user says "remember ...", use store_memory with label derived and content=verbatim memory.
- If user asks a question about a memory (e.g., 'Where is my passport?'), use recall_memory with question.
- For "add ..." with time like 'tomorrow 8am', put natural time in "when".
- For updates like 'change buy milk to 9am' use update_task with match="buy milk" and when="9am".
- For 'what are my tasks for today', use list_tasks with when="today".
Output ONLY the JSON. No commentary.
"""

def llm_to_json(user_msg: str) -> Dict[str, Any]:
    resp = groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user_msg}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    # try to extract JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"action":"help"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"action":"help"}

# --- Chat endpoint ---
@app.post("/chat")
def chat(req: ChatRequest):
    user_id = req.user_id
    cmd = llm_to_json(req.message)

    action = cmd.get("action","help")
    now_iso = req.now_iso
    tz = req.tz

    if action == "add_task":
        title = cmd.get("title") or cmd.get("match") or req.message
        description = cmd.get("description")
        when_text = cmd.get("when")
        due = parse_datetime(when_text, now=now_iso, tz=tz) if when_text else None
        task = add_task(user_id, title, description, due, req.message)
        confirm_time = f" at {due}" if due else ""
        return {"reply": f"✅ Added: “{task['title']}”{confirm_time}."}

    if action == "update_task":
        match = cmd.get("match") or cmd.get("title") or ""
        new_title = cmd.get("title") if cmd.get("title") and cmd.get("title") != match else None
        new_status = cmd.get("status")
        when_text = cmd.get("when")
        new_due = parse_datetime(when_text, now=now_iso, tz=tz) if when_text else None if when_text is not None else None
        task = update_task(user_id, match, new_title=new_title, new_due_at_iso=new_due, new_status=new_status)
        return {"reply": f"✏️ Updated: “{task['title']}”.", "task": task}

    if action == "delete_task":
        match = cmd.get("match") or cmd.get("title") or ""
        delete_task(user_id, match)
        return {"reply": f"🗑️ Deleted task matching “{match}”. (Most recent match removed)."}

    if action == "list_tasks":
        when = cmd.get("when")
        tasks = list_tasks(user_id, when=when)
        if not tasks: return {"reply":"No tasks found."}
        lines = []
        for t in tasks:
            due = t["due_at"]
            due_str = f" — {due}" if due else ""
            lines.append(f"• {t['title']}{due_str} [{t['status']}]")
        title = "Today’s tasks" if when == "today" else "Your tasks"
        return {"reply": f"**{title}**\n" + "\n".join(lines)}

    if action == "store_memory":
        # derive a label (simple heuristic)
        label = cmd.get("label")
        content = cmd.get("content") or cmd.get("description") or req.message
        if not label:
            # e.g., "wife favorite flower"
            label = "memory"
        m = store_memory(user_id, label, content)
        return {"reply": f"🧠 Saved memory: {label}."}

    if action == "recall_memory":
        question = cmd.get("question") or req.message
        label_contains = cmd.get("label")
        m = recall_memory(user_id, question, label_contains)
        if not m: return {"reply":"I couldn’t find a related memory."}
        return {"reply": f"🔎 Memory: {m['content']}"}

    # suggestions (simple LLM text)
    if action == "help":
        suggestion = groq.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role":"system","content":"Suggest helpful next actions for a personal task manager."},
                      {"role":"user","content":req.message}],
            temperature=0.5,
        ).choices[0].message.content
        return {"reply": suggestion}

@app.get("/health")
def health():
    return {"ok": True, "service": "remind-backend"}
