"""
Baseline Inference Script — SupportTriage OpenEnv
===================================================
Runs a GPT-4o-mini agent against all 3 tasks and reports reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline/run_baseline.py

    # Against a deployed environment:
    export OPENENV_BASE_URL=http://localhost:7860
    python baseline/run_baseline.py --mode remote

    # Local mode (no server needed):
    python baseline/run_baseline.py --mode local

Results are written to baseline/results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

import requests

# Add parent directory to path for local mode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Run: pip install openai")
    sys.exit(1)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL = "gpt-4o-mini"
TASKS = ["task_1_billing_duplicate", "task_2_technical_crash", "task_3_enterprise_escalation"]
RUNS_PER_TASK = 1  # Increase for variance analysis

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support agent. You handle support tickets by:
1. Triaging priority (low/medium/high/critical) and category (billing/technical/shipping/account/product/refund/general)
2. Gathering missing information from customers
3. Writing empathetic, professional responses
4. Escalating to the right team when needed
5. Resolving tickets efficiently

You interact with a structured environment. At each step you MUST output valid JSON matching the Action schema.

ACTION SCHEMA:
{
    "action_type": string,  // One of: triage, respond, escalate, resolve, request_info, apply_macro, internal_note, close
    
    // For triage:
    "priority": string,     // low | medium | high | critical
    "category": string,     // billing | technical | shipping | account | product | refund | general
    
    // For respond / request_info / internal_note:
    "message": string,
    
    // For escalate:
    "escalation_team": string,  // billing_specialist | senior_tech | refunds_team | account_security | management
    "escalation_reason": string,
    
    // For apply_macro:
    "macro_name": string,
    
    // For resolve / close:
    "resolution_summary": string
}

STRATEGY:
- Always triage FIRST (priority + category)
- For frustrated/angry customers: empathize immediately before asking questions
- For technical issues: request OS, app version, file details
- For billing: acknowledge and route to refunds team
- For enterprise customers with SLA breaches: escalate to management as CRITICAL
- Use internal_note to document context before escalating
- Resolve only after gathering info and responding/escalating
- Be concise but warm — aim for 40-80 word responses
""")


# ─────────────────────────────────────────────
# Local Mode (direct Python import)
# ─────────────────────────────────────────────

def run_local_episode(task_id: str, client: OpenAI) -> Dict[str, Any]:
    """Run an episode using the environment directly (no HTTP server)."""
    from env.environment import SupportTriageEnv
    from env.models import Action, ActionType

    env = SupportTriageEnv()
    obs = env.reset(task_id=task_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _obs_to_prompt(obs.dict())},
    ]

    total_reward = 0.0
    final_score = 0.0
    steps = []
    done = False

    while not done:
        # Get action from LLM
        action_dict = _llm_act(client, messages)
        if action_dict is None:
            break

        # Parse action
        try:
            action = Action(**action_dict)
        except Exception as e:
            print(f"  Action parse error: {e}")
            break

        # Step
        result = env.step(action)
        total_reward += result.reward.value
        done = result.done or result.truncated

        steps.append({
            "step": obs.step_number,
            "action": action_dict,
            "reward": result.reward.value,
            "reward_components": result.reward.components,
        })

        if done and "final_score" in result.info:
            final_score = result.info["final_score"]

        # Update conversation
        step_summary = _result_to_summary(action_dict, result.reward.dict(), result.done)
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        messages.append({"role": "user", "content": step_summary})

        obs = result.observation

    # Get final grade
    final_state = env.state()
    grade_result = final_state.grader_metadata or {}

    return {
        "task_id": task_id,
        "model": MODEL,
        "total_steps": final_state.step_number,
        "cumulative_reward": round(total_reward, 4),
        "final_score": final_score,
        "grade_breakdown": grade_result.get("breakdown", {}),
        "grade_diagnostics": grade_result.get("diagnostics", {}),
        "steps": steps,
    }


# ─────────────────────────────────────────────
# Remote Mode (HTTP API)
# ─────────────────────────────────────────────

def run_remote_episode(task_id: str, client: OpenAI, base_url: str) -> Dict[str, Any]:
    """Run an episode against the deployed HTTP server."""
    # Reset
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id})
    r.raise_for_status()
    data = r.json()
    episode_id = data["episode_id"]
    obs = data["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _obs_to_prompt(obs)},
    ]

    total_reward = 0.0
    final_score = 0.0
    steps = []
    done = False

    while not done:
        action_dict = _llm_act(client, messages)
        if action_dict is None:
            break

        r = requests.post(f"{base_url}/step", json={
            "episode_id": episode_id,
            "action": action_dict,
        })
        if r.status_code != 200:
            print(f"  Step error: {r.text}")
            break

        result = r.json()
        reward = result["reward"]["value"]
        total_reward += reward
        done = result["done"] or result["truncated"]

        steps.append({
            "step": result["observation"]["step_number"],
            "action": action_dict,
            "reward": reward,
            "reward_components": result["reward"].get("components", {}),
        })

        if done and "final_score" in result.get("info", {}):
            final_score = result["info"]["final_score"]

        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        messages.append({"role": "user", "content": _result_to_summary(
            action_dict, result["reward"], result["done"]
        )})

        obs = result["observation"]

    # Grade
    r = requests.post(f"{base_url}/grade?episode_id={episode_id}")
    grade_result = r.json().get("grade", {}) if r.status_code == 200 else {}

    return {
        "task_id": task_id,
        "model": MODEL,
        "total_steps": obs.get("step_number", 0),
        "cumulative_reward": round(total_reward, 4),
        "final_score": final_score or grade_result.get("score", 0.0),
        "grade_breakdown": grade_result.get("breakdown", {}),
        "grade_diagnostics": grade_result.get("diagnostics", {}),
        "steps": steps,
    }


# ─────────────────────────────────────────────
# LLM Helper
# ─────────────────────────────────────────────

def _llm_act(client: OpenAI, messages: List[Dict]) -> Optional[Dict]:
    """Call LLM and parse JSON action."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def _obs_to_prompt(obs: Dict) -> str:
    """Convert observation dict to LLM-friendly prompt."""
    ticket = obs.get("ticket", {})
    customer = ticket.get("customer", {})
    messages = ticket.get("messages", [])

    msg_text = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}"
        for m in messages
        if not m.get("is_internal", False)
    )

    return textwrap.dedent(f"""
    === SUPPORT TICKET ===
    Ticket ID: {ticket.get('ticket_id')}
    Subject: {ticket.get('subject')}
    Customer: {customer.get('name')} ({customer.get('account_tier')} tier, {customer.get('account_age_days')} days, {customer.get('previous_tickets')} prev tickets)
    Sentiment: {ticket.get('sentiment')}
    Order ID: {ticket.get('order_id', 'N/A')}
    Order Amount: ${ticket.get('order_amount', 0):.2f}

    === CONVERSATION ===
    {msg_text}

    === TASK ===
    {obs.get('task_description', '')}

    Step {obs.get('step_number')} / {obs.get('max_steps')}
    Available actions: {obs.get('available_actions', [])}
    Available macros: {obs.get('available_macros', [])}

    Respond with a JSON action object.
    """).strip()


def _result_to_summary(action: Dict, reward: Dict, done: bool) -> str:
    reward_val = reward.get("value", reward) if isinstance(reward, dict) else reward
    components = reward.get("components", {}) if isinstance(reward, dict) else {}
    return (
        f"Action taken: {action.get('action_type')}. "
        f"Reward: {reward_val:.3f} {components}. "
        f"{'Episode complete.' if done else 'Continue to next action.'}"
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baseline agent against SupportTriage OpenEnv")
    parser.add_argument("--mode", choices=["local", "remote"], default="local",
                        help="local: import env directly | remote: use HTTP API")
    parser.add_argument("--base-url", default=os.environ.get("OPENENV_BASE_URL", "http://localhost:7860"),
                        help="Base URL for remote mode")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Task IDs to run")
    parser.add_argument("--runs", type=int, default=RUNS_PER_TASK, help="Runs per task")
    parser.add_argument("--output", default="baseline/results.json", help="Output file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"\n{'='*60}")
    print(f"  SupportTriage OpenEnv — Baseline Evaluation")
    print(f"  Model: {MODEL}  |  Mode: {args.mode}  |  Runs/task: {args.runs}")
    print(f"{'='*60}\n")

    all_results = []
    task_scores = {}

    for task_id in args.tasks:
        print(f"\n📋 Task: {task_id}")
        print("-" * 40)
        task_run_scores = []

        for run in range(args.runs):
            print(f"  Run {run + 1}/{args.runs}...")
            start = time.time()

            if args.mode == "local":
                result = run_local_episode(task_id, client)
            else:
                result = run_remote_episode(task_id, client, args.base_url)

            elapsed = time.time() - start
            result["run"] = run + 1
            result["elapsed_seconds"] = round(elapsed, 2)
            all_results.append(result)
            task_run_scores.append(result["final_score"])

            print(f"    Score: {result['final_score']:.3f} | "
                  f"Reward: {result['cumulative_reward']:.3f} | "
                  f"Steps: {result['total_steps']} | "
                  f"Time: {elapsed:.1f}s")

            # Print grade breakdown
            if result.get("grade_breakdown"):
                for k, v in result["grade_breakdown"].items():
                    print(f"      {k}: {v:.3f}")

        avg = sum(task_run_scores) / len(task_run_scores) if task_run_scores else 0
        task_scores[task_id] = {"scores": task_run_scores, "mean": round(avg, 4)}
        print(f"\n  ✅ {task_id} mean score: {avg:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'='*60}")
    all_scores = []
    for task_id, ts in task_scores.items():
        difficulty = task_id.split("_")[1] if "_" in task_id else "?"
        print(f"  {task_id}: {ts['mean']:.3f}")
        all_scores.extend(ts["scores"])

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\n  OVERALL MEAN: {overall:.3f}")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "metadata": {
            "model": MODEL,
            "mode": args.mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runs_per_task": args.runs,
        },
        "task_scores": task_scores,
        "overall_mean": round(overall, 4),
        "episodes": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {args.output}")

    return output


if __name__ == "__main__":
    main()
