import os, json, re, datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from groq import Groq
import dateparser

# --- Env ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
groq = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Re:Mind Backend", version="1.0")

class ChatRequest(BaseModel):
    user_id: str
    message: str
    now_iso: Optional[str] = None
    tz: Optional[str] = None

SYSTEM = """You are Re:Mind's planner. Convert the user's message into a JSON command.
Schema:
{ "action":"add_todo|update_todo|delete_todo|list_todos|store_memory|recall_memory|help",
  "title":"...", "description":"...", "when":"...", "status":"open|done|cancelled",
  "match":"substring to find an existing todo", "label":"...", "question":"..."
}
Rules:
- "remember ..." -> store_memory (content is the memory; label optional).
- Questions about a past memory -> recall_memory (use question and/or label like 'passport').
- "add ..." with a time -> add_todo with when set to natural text (e.g., 'tomorrow 8am').
- "update buy milk -> 9am" -> update_todo with match='buy milk' and when='9am'.
- "what are my tasks for today" -> list_todos with when='today'.
Output ONLY the JSON object.
"""

def parse_datetime(text: str, now_iso: Optional[str], tz: Optional[str]) -> Optional[str]:
    if not text: return None
    settings = {"PREFER_DATES_FROM": "future"}
    if tz: settings["TIMEZONE"] = tz
    base = None
    if now_iso:
        try:
            # make a naive ISO acceptable
            if now_iso.endswith("Z"):
                now_iso = now_iso.replace("Z","+00:00")
            base = datetime.datetime.fromisoformat(now_iso)
        except Exception:
            base = None
    dt = dateparser.parse(text, settings=settings) if base is None else dateparser.parse(text, settings=settings, RELATIVE_BASE=base)
    if not dt: return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    dt_utc = dt.astimezone(datetime.timezone.utc).replace(microsecond=0)
    return dt_utc.isoformat().replace("+00:00","Z")

def llm_to_json(user_msg: str) -> Dict[str, Any]:
    resp = groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":user_msg}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m: return {"action":"help"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"action":"help"}

# --- DB helpers using YOUR schema (public.todos, public.memories) ---
def todos_list(user_id: str, when: Optional[str]) -> List[Dict[str,Any]]:
    q = supabase.table("todos").select("*").eq("user_id", user_id)
    if when == "today":
        today = datetime.datetime.utcnow().date()
        start = datetime.datetime.combine(today, datetime.time.min, tzinfo=datetime.timezone.utc).isoformat().replace("+00:00","Z")
        end   = datetime.datetime.combine(today, datetime.time.max, tzinfo=datetime.timezone.utc).isoformat().replace("+00:00","Z")
        q = q.gte("due_date", start).lte("due_date", end)
    return q.order("due_date", desc=False).execute().data or []

def todos_add(user_id: str, title: str, due_iso: Optional[str], source: str) -> Dict[str,Any]:
    row = {
        "user_id": user_id,
        "task": title.strip(),
        "due_date": due_iso,
        "is_completed": False,
        # reminded defaults to false
    }
    res = supabase.table("todos").insert(row).execute()
    return res.data[0]

def todos_update(user_id: str, match: str, new_title: Optional[str], new_due_iso: Optional[str], new_status: Optional[str]) -> Dict[str,Any]:
    # pull most recent matching
    qs = supabase.table("todos").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
    cand = next((t for t in qs if match.lower() in t["task"].lower()), None)
    if not cand: raise HTTPException(404, "Todo not found")
    update = {}
    if new_title: update["task"] = new_title
    if new_status:
        if new_status == "done": update["is_completed"] = True
        elif new_status == "cancelled": update["is_completed"] = True
        else: update["is_completed"] = False
    if new_due_iso is not None:
        update["due_date"] = new_due_iso
        update["reminded"] = False   # reset reminder if date changed
        update["reminded_at"] = None
    if update:
        supabase.table("todos").update(update).eq("id", cand["id"]).execute()
    return supabase.table("todos").select("*").eq("id", cand["id"]).single().execute().data

def todos_delete(user_id: str, match: str) -> int:
    qs = supabase.table("todos").select("id,task").eq("user_id", user_id).order("created_at", desc=True).execute().data
    cand = next((t for t in qs if match.lower() in t["task"].lower()), None)
    if not cand: raise HTTPException(404, "Todo not found")
    supabase.table("todos").delete().eq("id", cand["id"]).execute()
    return 1

def memory_store(user_id: str, content: str) -> Dict[str,Any]:
    res = supabase.table("memories").insert({"user_id": user_id, "content": content}).execute()
    return res.data[0]

def memory_recall(user_id: str, question: Optional[str]) -> Optional[Dict[str,Any]]:
    rows = supabase.table("memories").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
    if not rows: return None
    if question:
        for m in rows:
            if question.lower() in (m["content"] or "").lower():
                return m
    return rows[0]

@app.post("/chat")
def chat(req: ChatRequest):
    cmd = llm_to_json(req.message)
    action = cmd.get("action","help")

    if action == "add_todo":
        title = cmd.get("title") or cmd.get("match") or req.message
        due = parse_datetime(cmd.get("when"), req.now_iso, req.tz) if cmd.get("when") else None
        t = todos_add(req.user_id, title, due, req.message)
        return {"reply": f"✅ Added: “{t['task']}”" + (f" at {t['due_date']}" if t['due_date'] else "")}

    if action == "update_todo":
        match = cmd.get("match") or cmd.get("title") or ""
        new_title = cmd.get("title") if cmd.get("title") and cmd.get("title") != match else None
        new_due = parse_datetime(cmd.get("when"), req.now_iso, req.tz) if "when" in cmd else None
        new_status = cmd.get("status")
        t = todos_update(req.user_id, match, new_title, new_due, new_status)
        return {"reply": f"✏️ Updated: “{t['task']}”."}

    if action == "delete_todo":
        match = cmd.get("match") or cmd.get("title") or ""
        todos_delete(req.user_id, match)
        return {"reply": f"🗑️ Deleted todo matching “{match}”. (Most recent match removed)."}

    if action == "list_todos":
        when = cmd.get("when")
        items = todos_list(req.user_id, when)
        if not items: return {"reply":"No todos found."}
        title = "Today’s tasks" if when == "today" else "Your tasks"
        lines = []
        for t in items:
            due = f" — {t['due_date']}" if t["due_date"] else ""
            status = "done" if t["is_completed"] else "open"
            lines.append(f"• {t['task']}{due} [{status}]")
        return {"reply": f"**{title}**\n" + "\n".join(lines)}

    if action == "store_memory":
        content = cmd.get("description") or cmd.get("content") or req.message
        memory_store(req.user_id, content)
        return {"reply": "🧠 Saved that to memory."}

    if action == "recall_memory":
        question = cmd.get("question") or req.message
        m = memory_recall(req.user_id, question)
        return {"reply": f"🔎 {m['content']}" if m else "I couldn’t find a related memory."}

    suggestion = groq.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role":"system","content":"Suggest helpful next actions for a personal task manager."},
                  {"role":"user","content":req.message}],
        temperature=0.5,
    ).choices[0].message.content
    return {"reply": suggestion}

@app.get("/health")
def health():
    return {"ok": True}
