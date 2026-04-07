"""
FastAPI server implementing the OpenEnv HTTP interface.
Runs on port 7860 for Hugging Face Spaces compatibility.

Endpoints:
  GET  /                   -> Environment info & welcome
  GET  /openenv.yaml       -> Environment metadata
  GET  /tasks              -> List all tasks
  POST /reset              -> Start new episode
  POST /step               -> Take a step
  GET  /state              -> Get current state
  POST /grade              -> Grade current episode
  GET  /health             -> Health check
  GET  /docs               -> FastAPI auto-docs
"""

from __future__ import annotations

import os
import yaml
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from env.environment import SupportTriageEnv
from env.models import Action, ActionType
from tasks.task_definitions import list_tasks, TASKS
from graders.task_graders import grade

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="SupportTriage OpenEnv",
    description=(
        "Customer Support Triage & Resolution Environment. "
        "An AI agent handles realistic support tickets: triaging priority, "
        "gathering information, de-escalating angry customers, and resolving cases."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (episode_id -> env instance)
# For production, use Redis or similar
SESSIONS: Dict[str, SupportTriageEnv] = {}

OPENENV_YAML_CONTENT = {
    "name": "support-triage-v1",
    "version": "1.0.0",
    "description": (
        "Customer Support Triage & Resolution: An AI agent handles realistic customer "
        "support tickets ranging from billing disputes to angry enterprise escalations. "
        "Tasks require triaging correctly, gathering diagnostic information, empathetic "
        "de-escalation, and efficient resolution."
    ),
    "tags": ["openenv", "customer-support", "triage", "nlp", "real-world", "rl"],
    "author": "openenv-support-triage",
    "tasks": [
        {
            "id": "task_1_billing_duplicate",
            "difficulty": "easy",
            "description": "Duplicate billing charge — triage, acknowledge, request info, escalate to refunds",
            "max_steps": 6,
        },
        {
            "id": "task_2_technical_crash",
            "difficulty": "medium",
            "description": "App crash with missing diagnostics — empathize, gather info, offer workaround, escalate",
            "max_steps": 8,
        },
        {
            "id": "task_3_enterprise_escalation",
            "difficulty": "hard",
            "description": "Furious enterprise customer, SLA breach, policy override, legal threat — de-escalate & resolve",
            "max_steps": 10,
        },
    ],
    "observation_space": {
        "type": "dict",
        "fields": {
            "ticket": "Ticket object with messages, customer profile, metadata",
            "queue_stats": "Queue-level stats (total open, SLA breaches, etc.)",
            "available_macros": "List of macro template names",
            "available_actions": "List of currently valid ActionTypes",
            "step_number": "Current step in episode",
            "max_steps": "Max steps allowed",
            "task_description": "Natural language task description",
        },
    },
    "action_space": {
        "type": "dict",
        "action_types": [
            "triage", "respond", "escalate", "resolve",
            "request_info", "apply_macro", "internal_note", "close"
        ],
        "description": "Typed actions with optional fields depending on action_type",
    },
    "reward_range": [-1.0, 1.0],
    "reward_description": (
        "Dense reward signal: triage accuracy (+0.15/-0.10), response quality (0.05-0.10), "
        "escalation accuracy (+0.15/-0.05), resolution bonus (+0.20), "
        "efficiency bonus (+0.05), empathy bonus (+0.05), loop penalty (-0.08/repeat)"
    ),
    "endpoints": {
        "reset": "POST /reset",
        "step": "POST /step",
        "state": "GET /state",
        "tasks": "GET /tasks",
    },
}


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    episode_id: str
    action: Action


class StateRequest(BaseModel):
    episode_id: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    tasks_html = "".join(
        f"<li><strong>{t['task_id']}</strong> [{t['difficulty']}] — {t['description']}</li>"
        for t in list_tasks()
    )
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SupportTriage OpenEnv</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #0f1117; color: #e8e8e8; }}
            h1 {{ color: #7ee8a2; letter-spacing: -1px; }}
            h2 {{ color: #7ec8e3; border-bottom: 1px solid #333; padding-bottom: 8px; }}
            pre {{ background: #1a1d27; padding: 16px; border-radius: 8px; overflow-x: auto; color: #a8d8a8; }}
            a {{ color: #7ec8e3; }}
            .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 6px; }}
            .easy {{ background: #1a4a1a; color: #7ee8a2; }}
            .medium {{ background: #4a3a1a; color: #e8c47a; }}
            .hard {{ background: #4a1a1a; color: #e87a7a; }}
            ul {{ line-height: 2; }}
            .endpoint {{ font-family: monospace; background: #1a1d27; padding: 2px 6px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>🎯 SupportTriage OpenEnv v1.0.0</h1>
        <p>
            A real-world <strong>Customer Support Triage &amp; Resolution</strong> environment 
            for training and evaluating AI agents on genuine business tasks.
        </p>

        <h2>Tasks</h2>
        <ul>{tasks_html}</ul>

        <h2>Quick Start</h2>
        <pre>
# 1. Start a new episode
POST /reset
{{"task_id": "task_1_billing_duplicate"}}

# 2. Take a step
POST /step
{{
  "episode_id": "&lt;from reset response&gt;",
  "action": {{
    "action_type": "triage",
    "priority": "medium",
    "category": "billing"
  }}
}}

# 3. Get state
GET /state?episode_id=&lt;episode_id&gt;</pre>

        <h2>Links</h2>
        <ul>
            <li><a href="/docs">📖 Interactive API Docs (Swagger)</a></li>
            <li><a href="/openenv.yaml">📄 openenv.yaml</a></li>
            <li><a href="/tasks">📋 Task List (JSON)</a></li>
            <li><a href="/health">💚 Health Check</a></li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "support-triage-v1", "version": "1.0.0", "timestamp": time.time()}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
async def get_openenv_yaml():
    return yaml.dump(OPENENV_YAML_CONTENT, default_flow_style=False, sort_keys=False)


@app.get("/tasks")
async def get_tasks():
    return {"tasks": list_tasks()}


@app.get("/tasks/{task_id}")
async def get_task_detail(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    meta = TASKS[task_id]["meta"].copy()
    # Remove internal grader keys from public response
    meta.pop("correct_escalation", None)
    meta.pop("secondary_escalation", None)
    return meta


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    task_id = request.task_id if request and request.task_id else "task_1_billing_duplicate"
    env = SupportTriageEnv()
    try:
        obs = env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    episode_id = obs.episode_id
    SESSIONS[episode_id] = env

    # Prune old sessions (keep latest 100)
    if len(SESSIONS) > 100:
        oldest = list(SESSIONS.keys())[0]
        del SESSIONS[oldest]

    return {
        "episode_id": episode_id,
        "observation": obs.dict(),
        "message": f"Episode started for task: {task_id}",
    }


@app.post("/step")
async def step(request: StepRequest):
    env = SESSIONS.get(request.episode_id)
    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"No active session for episode_id: {request.episode_id}. Call /reset first.",
        )

    try:
        result = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": result.observation.dict(),
        "reward": result.reward.dict(),
        "done": result.done,
        "truncated": result.truncated,
        "info": result.info,
    }


@app.get("/state")
async def state(episode_id: str):
    env = SESSIONS.get(episode_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"No active session: {episode_id}")
    return env.state().dict()


@app.post("/grade")
async def grade_episode(episode_id: str):
    env = SESSIONS.get(episode_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"No active session: {episode_id}")
    state = env.state()
    result = grade(state.task_id, state)
    return {"episode_id": episode_id, "task_id": state.task_id, "grade": result}


def run_server() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_server()
