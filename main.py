# ====== 1. IMPORTS AND SETUP ======
import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client, Client

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

from langchain_openai import OpenAIEmbeddings


# ====== 2. CONFIGURE API SERVER ======
app = FastAPI(title="Re:Mind Backend")

# Minimal, correct CORS: wildcard origins only if no credentials.
# In production, set FRONTEND_ORIGIN to your Vercel URL and switch to that.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "")
allowed_origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,                 # keep False if using "*"
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ====== 3. INITIALIZE SERVICES ======
# Load API keys from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # service role key (server-side only)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")  # ensure you set this in Render

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    # Keep your original Groq model string (works for you)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=GROQ_API_KEY)

    # Keep OpenAIEmbeddings (1536-dim). Be explicit on model to avoid surprises.
    # text-embedding-3-small -> 1536 dims
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
except Exception as e:
    raise RuntimeError(f"Failed to initialize external services: {e}")


# Pydantic model for incoming API requests
class UserInput(BaseModel):
    user_id: str
    message: str


# ====== 4. DEFINE AGENT TOOLS (SKILLS) ======
@tool
def add_task(user_id: str, task_description: str, due_date: Optional[str] = None):
    """Adds a new to-do item. Requires user_id and task_description. Optional due_date (ISO)."""
    try:
        todo_item = {"task": task_description, "user_id": user_id, "is_completed": False}
        if due_date:
            iso = due_date.replace("Z", "+00:00")
            todo_item["due_date"] = datetime.fromisoformat(iso).isoformat()

        response = supabase.table('todos').insert(todo_item).execute()
        return "Successfully added task." if response.data else f"Error adding task: {response.error}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_tasks(user_id: str):
    """Retrieves all of a user's active to-do items. Requires user_id."""
    try:
        response = (
            supabase.table('todos')
            .select('id, task, due_date')
            .eq('user_id', user_id)
            .eq('is_completed', False)
            .order('due_date', desc=False)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return "You have no active tasks."
        formatted = []
        for r in rows:
            due = r.get("due_date")
            due_txt = f" (due {due})" if due else ""
            formatted.append(f"- '{r['task']}' (ID: {r['id']}){due_txt}")
        return "Current tasks:\n" + "\n".join(formatted)
    except Exception as e:
        return f"Error: {e}"


@tool
def update_task(
    user_id: str,
    task_id: int,
    new_description: Optional[str] = None,
    new_due_date: Optional[str] = None,
    is_completed: Optional[bool] = None
):
    """Updates an existing to-do item. Requires user_id and task_id."""
    try:
        updates = {}
        if new_description is not None:
            updates["task"] = new_description
        if new_due_date is not None:
            iso = new_due_date.replace("Z", "+00:00")
            updates["due_date"] = datetime.fromisoformat(iso).isoformat()
        if is_completed is not None:
            updates["is_completed"] = is_completed

        if not updates:
            return "Error: No update information provided."

        response = (
            supabase.table('todos')
            .update(updates)
            .eq('user_id', user_id)
            .eq('id', task_id)
            .execute()
        )
        return "Task updated successfully." if response.data else "Failed to update task. Check task_id."
    except Exception as e:
        return f"Error: {e}"


@tool
def delete_task(user_id: str, task_id: int):
    """Deletes a to-do item. Requires user_id and task_id. Use get_tasks to find the task_id first."""
    try:
        response = (
            supabase.table('todos')
            .delete()
            .eq('user_id', user_id)
            .eq('id', task_id)
            .execute()
        )
        return "Task deleted successfully." if response.data else "Failed to delete task. Check task_id."
    except Exception as e:
        return f"Error: {e}"


@tool
def store_memory(user_id: str, text_content: str):
    """Stores a piece of information as a memory. For remembering facts, not tasks."""
    try:
        embedding = embeddings.embed_query(text_content)  # 1536-dim vector
        response = (
            supabase.table('memories')
            .insert({"content": text_content, "embedding": embedding, "user_id": user_id})
            .execute()
        )
        return "OK, I've stored that memory." if response.data else f"Error storing memory: {response.error}"
    except Exception as e:
        return f"Error: {e}"


@tool
def recall_memory(user_id: str, search_query: str):
    """Searches memories to answer a user's question."""
    try:
        qvec = embeddings.embed_query(search_query)  # 1536-dim
        # Minimal, correct RPC call: arguments must match the SQL function signature; no .eq() chaining
        res = supabase.rpc(
            'match_memories',
            {'user_id': user_id, 'query_embedding': qvec, 'match_count': 1}
        ).execute()
        rows = res.data or []
        return f"Found a relevant memory: '{rows[0]['content']}'" if rows else \
               "I don't have a memory that matches that question."
    except Exception as e:
        return f"Error: {e}"


# ====== 5. CREATE AGENT AND EXECUTOR ======
tools = [add_task, get_tasks, update_task, delete_task, store_memory, recall_memory]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Re:Mind, a helpful assistant. "
               "If text is a memory (e.g., 'remember ...'), call store_memory. "
               "If it's a task (add/read/update/delete), call the appropriate tool. "
               "Infer natural-language dates when you can; if unsure, ask briefly. "
               "Never invent task IDs—list first to identify."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ====== 6. DEFINE API ENDPOINTS ======
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Re:Mind backend is running"}


@app.post("/invoke", tags=["Agent"])
async def invoke_agent(user_input: UserInput):
    """Main endpoint to process user messages via the LangChain agent."""
    if not user_input.user_id or not user_input.message:
        raise HTTPException(status_code=400, detail="user_id and message are required.")

    input_with_context = f"My user_id is '{user_input.user_id}'. The user's request is: '{user_input.message}'"

    try:
        response = agent_executor.invoke({"input": input_with_context})
        text_output = response.get('output', 'Agent failed to produce a final output.')
        # Keep response shape consistent with the frontend expectation
        return {"reply": text_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {e}")
