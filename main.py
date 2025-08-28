# ====== 1. IMPORTS AND SETUP ======
import os
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

# ====== 2. INITIALIZE SERVICES ======
# Load API keys from environment variables set in the hosting service.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")

# Establish connections to Supabase, Groq LLM, and the Embeddings model.
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=GROQ_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize external services: {e}")

# Pydantic model for validating incoming API requests.
class UserInput(BaseModel):
    user_id: str
    message: str

# ====== 3. DEFINE AGENT TOOLS (SKILLS) ======
# These are the functions the AI agent can execute to perform actions.

@tool
def add_task(user_id: str, task_description: str, due_date: Optional[str] = None):
    """Adds a new to-do item. Requires user_id and task_description. Optional due_date."""
    try:
        todo_item = {"task": task_description, "user_id": user_id, "is_completed": False}
        if due_date:
            todo_item["due_date"] = datetime.fromisoformat(due_date.replace("Z", "+00:00")).isoformat()
        response = supabase.table('todos').insert(todo_item).execute()
        return f"Successfully added task." if response.data else f"Error adding task: {response.error}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_tasks(user_id: str):
    """Retrieves all of a user's active to-do items. Requires user_id."""
    try:
        response = supabase.table('todos').select('id, task').eq('user_id', user_id).eq('is_completed', False).execute()
        if response.data:
            formatted_tasks = [f"- '{item['task']}' (ID: {item['id']})" for item in response.data]
            return "Current tasks:\n" + "\n".join(formatted_tasks)
        return "You have no active tasks."
    except Exception as e:
        return f"Error: {e}"

@tool
def update_task(user_id: str, task_id: str, new_description: Optional[str] = None, new_due_date: Optional[str] = None, is_completed: Optional[bool] = None):
    """Updates an existing to-do item. Requires user_id and task_id. Use get_tasks to find the task_id first."""
    try:
        updates = {k: v for k, v in locals().items() if k in ['new_description', 'new_due_date', 'is_completed'] and v is not None}
        if not updates: return "Error: No update information provided."
        
        if 'new_description' in updates: updates['task'] = updates.pop('new_description')
        if 'new_due_date' in updates: updates['due_date'] = datetime.fromisoformat(updates.pop('new_due_date').replace("Z", "+00:00")).isoformat()
        
        response = supabase.table('todos').update(updates).eq('user_id', user_id).eq('id', task_id).execute()
        return "Task updated successfully." if response.data else "Failed to update task. Check task_id."
    except Exception as e:
        return f"Error: {e}"

@tool
def delete_task(user_id: str, task_id: str):
    """Deletes a to-do item. Requires user_id and task_id. Use get_tasks to find the task_id first."""
    try:
        response = supabase.table('todos').delete().eq('user_id', user_id).eq('id', task_id).execute()
        return "Task deleted successfully." if response.data else "Failed to delete task. Check task_id."
    except Exception as e:
        return f"Error: {e}"

@tool
def store_memory(user_id: str, text_content: str):
    """Stores a piece of information as a memory. For remembering facts, not tasks."""
    try:
        embedding = embeddings.embed_query(text_content)
        response = supabase.table('memories').insert({"content": text_content, "embedding": embedding, "user_id": user_id}).execute()
        return "OK, I've stored that memory." if response.data else f"Error storing memory: {response.error}"
    except Exception as e:
        return f"Error: {e}"

@tool
def recall_memory(user_id: str, search_query: str):
    """Searches memories to answer a user's question."""
    try:
        embedding = embeddings.embed_query(search_query)
        matches = supabase.rpc('match_memories', {'query_embedding': embedding, 'match_threshold': 0.75, 'match_count': 1}).eq('user_id', user_id).execute()
        return f"Found a relevant memory: '{matches.data[0]['content']}'" if matches.data else "I don't have a memory that matches that question."
    except Exception as e:
        return f"Error: {e}"

# ====== 4. CREATE AGENT AND EXECUTOR ======
tools = [add_task, get_tasks, update_task, delete_task, store_memory, recall_memory]

# The agent's core instructions and personality.
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Re:Mind, a helpful assistant. Use your tools to manage the user's to-do lists and memories. For updates or deletes, you MUST use get_tasks first to find the correct task_id."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ====== 5. CONFIGURE API SERVER ======
app = FastAPI(title="Re:Mind Backend")

@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Re:Mind backend is running"}

@app.post("/invoke", tags=["Agent"])
async def invoke_agent(user_input: UserInput):
    """Main endpoint to process user messages via the LangChain agent."""
    if not all([user_input.user_id, user_input.message]):
        raise HTTPException(status_code=400, detail="user_id and message are required.")
    
    # Provide the agent with necessary context for tool usage.
    input_with_context = f"My user_id is '{user_input.user_id}'. The user's request is: '{user_input.message}'"
    
    try:
        response = agent_executor.invoke({"input": input_with_context})
        return {"output": response.get('output', 'Agent failed to produce a final output.')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {e}")
