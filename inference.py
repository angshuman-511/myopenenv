"""
Minimal baseline inference entrypoint required by OpenEnv submission checks.

This script exposes a deterministic policy via `infer_action` and can also run
an end-to-end episode against a deployed OpenEnv HTTP API.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import requests


def _task_defaults(task_id: str) -> Dict[str, str]:
    if task_id == "task_1_billing_duplicate":
        return {"priority": "medium", "category": "billing"}
    if task_id == "task_2_technical_crash":
        return {"priority": "high", "category": "technical"}
    if task_id == "task_3_enterprise_escalation":
        return {"priority": "critical", "category": "refund"}
    return {"priority": "medium", "category": "general"}


def _allowed(observation: Dict[str, Any], action_type: str) -> bool:
    allowed_actions = observation.get("available_actions", [])
    return action_type in allowed_actions


def infer_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Return the next action dict from an observation."""
    task_id = observation.get("task_id", "")
    step = int(observation.get("step_number", 0))
    defaults = _task_defaults(task_id)

    # Start with triage if ticket is not triaged yet.
    ticket = observation.get("ticket", {})
    if (ticket.get("priority") is None or ticket.get("category") is None) and _allowed(observation, "triage"):
        return {
            "action_type": "triage",
            "priority": defaults["priority"],
            "category": defaults["category"],
        }

    if task_id == "task_1_billing_duplicate":
        if step <= 1 and _allowed(observation, "respond"):
            return {
                "action_type": "respond",
                "message": "I am sorry about the duplicate charge. I understand the frustration and I will help resolve this quickly.",
            }
        if step <= 2 and _allowed(observation, "request_info"):
            return {
                "action_type": "request_info",
                "message": "Please share the transaction date and the last 4 digits on your statement so I can verify the duplicate payment.",
            }
        if step <= 3 and _allowed(observation, "escalate"):
            return {
                "action_type": "escalate",
                "escalation_team": "refunds_team",
                "escalation_reason": "Verified duplicate billing charge requiring refund processing.",
            }

    elif task_id == "task_2_technical_crash":
        if step <= 1 and _allowed(observation, "respond"):
            return {
                "action_type": "respond",
                "message": "I am sorry this happened before your deadline. We can use a workaround while we investigate the crash.",
            }
        if step <= 2 and _allowed(observation, "request_info"):
            return {
                "action_type": "request_info",
                "message": "Please share your OS, app version, and approximate PDF file size so we can reproduce the issue.",
            }
        if step <= 3 and _allowed(observation, "escalate"):
            return {
                "action_type": "escalate",
                "escalation_team": "senior_tech",
                "escalation_reason": "Export crash with urgency and missing diagnostics.",
            }

    elif task_id == "task_3_enterprise_escalation":
        if step <= 1 and _allowed(observation, "respond"):
            return {
                "action_type": "respond",
                "message": "I am truly sorry for the delays and repeated follow-up failures. I own this and I will coordinate an urgent recovery plan now.",
            }
        if step <= 2 and _allowed(observation, "internal_note"):
            return {
                "action_type": "internal_note",
                "message": "Enterprise escalation: SLA breach, high-value order, legal risk. Prioritize immediate callback and refund coordination.",
            }
        if step <= 3 and _allowed(observation, "escalate"):
            return {
                "action_type": "escalate",
                "escalation_team": "management",
                "escalation_reason": "Enterprise SLA breach and legal escalation risk.",
            }

    if _allowed(observation, "resolve"):
        return {
            "action_type": "resolve",
            "resolution_summary": "Triage completed, customer acknowledged, and appropriate follow-up actions were initiated.",
        }

    if _allowed(observation, "respond"):
        return {
            "action_type": "respond",
            "message": "Thank you for your patience. I am actively working on this and will share the next update shortly.",
        }

    # Fallback for unexpected action constraints.
    return {"action_type": "close", "resolution_summary": "Unable to continue due to action constraints."}


def run_episode(base_url: str, task_id: str) -> Dict[str, Any]:
    reset_resp = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()

    episode_id = reset_data["episode_id"]
    observation = reset_data["observation"]
    done = False
    truncated = False
    steps = 0

    while not done and not truncated and steps < 20:
        action = infer_action(observation)
        step_resp = requests.post(
            f"{base_url}/step",
            json={"episode_id": episode_id, "action": action},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()
        observation = step_data["observation"]
        done = bool(step_data["done"])
        truncated = bool(step_data["truncated"])
        steps += 1

    grade_resp = requests.post(f"{base_url}/grade?episode_id={episode_id}", timeout=30)
    grade_payload = grade_resp.json() if grade_resp.ok else {"grade": {}}
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "steps": steps,
        "done": done,
        "truncated": truncated,
        "grade": grade_payload.get("grade", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic baseline inference for SupportTriage OpenEnv.")
    parser.add_argument("--base-url", default=os.getenv("OPENENV_BASE_URL", "http://localhost:7860"))
    parser.add_argument("--task-id", default="task_1_billing_duplicate")
    args = parser.parse_args()

    summary = run_episode(args.base_url, args.task_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
