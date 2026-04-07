# 🎯 SupportTriage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-7ee8a2)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-support--triage--v1-blue)](https://huggingface.co/spaces/openenv/support-triage-v1)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

A real-world **Customer Support Triage & Resolution** environment for training and evaluating AI agents. Agents handle realistic support tickets — from simple billing disputes to furious enterprise customers threatening legal action — and must demonstrate triage accuracy, empathetic communication, diagnostic reasoning, and policy judgment.

---

## 🏭 Why This Environment?

Customer support is one of the highest-leverage real-world applications of LLM agents:

- **Scale**: Companies handle millions of tickets per month; even marginal agent improvement has enormous economic value
- **Complexity**: Requires multi-step reasoning, empathy calibration, policy knowledge, and escalation judgment
- **Measurability**: Success is concrete — correct priority/category, right team escalated, customer de-escalated, issue resolved
- **Gap**: No existing OpenEnv covers the full support workflow from triage → diagnosis → response → escalation → resolution

This environment fills that gap with three tasks spanning easy → hard that meaningfully challenge frontier models.

---

## 🗂 Project Structure

```
openenv-support-triage/
├── app.py                    # FastAPI server (OpenEnv HTTP interface)
├── openenv.yaml              # Environment metadata spec
├── Dockerfile                # Container for HF Spaces deployment
├── requirements.txt
├── env/
│   ├── models.py             # Typed Pydantic models (Observation, Action, Reward, ...)
│   └── environment.py        # SupportTriageEnv: reset() / step() / state()
├── tasks/
│   └── task_definitions.py   # 3 task scenarios with metadata
├── graders/
│   └── task_graders.py       # Deterministic graders (0.0–1.0) for each task
├── baseline/
│   └── run_baseline.py       # OpenAI API baseline inference script
└── tests/
    ├── test_environment.py   # Full pytest test suite
    └── validate_logic.py     # Standalone validation (no external deps)
```

---

## 📋 Tasks

### Task 1 — Duplicate Billing Charge *(Easy)*
**Max Steps:** 6 | **Correct Priority:** MEDIUM | **Correct Category:** BILLING

A verified Basic-tier customer reports being charged twice ($29.99) for their subscription and provides an order number. The customer is polite and straightforward.

**Agent must:**
- Triage correctly (BILLING / MEDIUM)
- Acknowledge the issue with empathy
- Request transaction confirmation details before processing
- Escalate to the refunds team

**Why it's easy:** Single clear intent, cooperative customer, obvious resolution path, all info present.

**Scoring rubric:**

| Component | Weight | Criteria |
|-----------|--------|----------|
| Correct triage | 25% | Priority=medium AND category=billing |
| Customer acknowledged | 20% | Empathetic language in response |
| Info requested or escalated | 25% | Asked for transaction details OR escalated to refunds |
| Resolution quality | 30% | Relevant content + resolved/escalated + message quality |

**Baseline score (gpt-4o-mini):** `0.72`

---

### Task 2 — App Crash with Missing Diagnostics *(Medium)*
**Max Steps:** 8 | **Correct Priority:** HIGH | **Correct Category:** TECHNICAL

A frustrated Premium-tier customer's app crashes on PDF export with a presentation due the next morning. Critical diagnostic information is absent: OS, app version, file size/type. The customer has already tried restarting.

**Agent must:**
- Triage correctly (TECHNICAL / HIGH — premium tier + urgency)
- Empathize with the time pressure ("presentation tomorrow")
- Request all three diagnostic categories: OS, app version, file details
- Offer a practical workaround while the issue is investigated
- Escalate to senior tech support

**Why it's medium:** Requires inference (premium = high priority), missing info diagnosis, multi-part response, workaround creativity.

**Scoring rubric:**

| Component | Weight | Criteria |
|-----------|--------|----------|
| Correct triage | 20% | Priority=high AND category=technical |
| Empathy for urgency | 15% | References presentation/tomorrow/urgency |
| Diagnostic info requested | 25% | Asks for OS + version + file (each 1/3) |
| Workaround offered | 15% | Suggests alternative export format/method |
| Escalation to correct team | 25% | Escalates to senior_tech |

**Baseline score (gpt-4o-mini):** `0.58`

---

### Task 3 — Enterprise Escalation: SLA Breach + Legal Threat *(Hard)*
**Max Steps:** 10 | **Correct Priority:** CRITICAL | **Correct Category:** REFUND

An Enterprise customer ($4,800 order, 730-day account) is furious. Wrong product delivered three weeks ago, four failed contact attempts, promises broken by "Alex," and now threatening chargebacks and legal action. The agent must handle this while recognizing a policy edge case: refunds over $500 normally require manager approval, but enterprise tier + SLA breach constitutes an override.

**Agent must:**
- Triage as CRITICAL/REFUND immediately
- Own the failure without deflecting ("our policy says..." is penalized)
- De-escalate with genuine empathy and personal commitment language
- Escalate to MANAGEMENT (not just refunds team)
- Make a specific, time-bound callback commitment
- Add a substantive internal note (≥20 words) documenting the situation
- Recognize the enterprise SLA policy override (in note or response)

**Why it's hard:** Requires anger de-escalation, policy reasoning, multi-team coordination, balancing apology vs. commitment specificity, internal documentation.

**Scoring rubric:**

| Component | Weight | Criteria |
|-----------|--------|----------|
| Correct triage (critical) | 15% | Priority=critical AND category=refund |
| Acknowledgment without deflection | 20% | Owns failure; penalized for "per policy" deflection |
| De-escalation quality | 20% | Empathy density score across 10 keywords |
| Escalation to management | 15% | Escalates to management team |
| Concrete commitment | 15% | Specific timeframe + callback promised |
| Internal note added | 10% | Substantive internal note (≥20 words) |
| Policy override recognized | 5% | Acknowledges enterprise/SLA exception |

**Baseline score (gpt-4o-mini):** `0.38`

---

## 🔧 Action Space

All actions are typed JSON objects with an `action_type` field and conditional required fields:

```json
// Triage (must be first meaningful action)
{"action_type": "triage", "priority": "medium", "category": "billing"}

// Respond to customer
{"action_type": "respond", "message": "Thank you for reaching out..."}

// Request missing information
{"action_type": "request_info", "message": "Could you please provide your OS version..."}

// Apply pre-written macro
{"action_type": "apply_macro", "macro_name": "request_diagnostic_info"}

// Add internal note (not visible to customer)
{"action_type": "internal_note", "message": "Enterprise customer, SLA breached 3 weeks..."}

// Escalate to specialist team
{
  "action_type": "escalate",
  "escalation_team": "management",
  "escalation_reason": "Enterprise SLA breach, legal threat"
}

// Resolve ticket
{"action_type": "resolve", "resolution_summary": "Duplicate charge refunded via refunds team."}

// Close (penalized — tickets in this env are never spam)
{"action_type": "close"}
```

**Priority values:** `low | medium | high | critical`  
**Category values:** `billing | technical | shipping | account | product | refund | general`  
**Escalation teams:** `billing_specialist | senior_tech | refunds_team | account_security | management`

**Available macros:**
- `greeting_standard` — standard welcome
- `billing_refund_initiated` — refund confirmation
- `technical_troubleshooting_steps` — diagnostic request
- `request_order_number` — order number request
- `request_diagnostic_info` — OS/version/file request
- `escalation_acknowledgment` — escalation notice
- `resolution_confirmed` — resolution confirmation
- `apology_delay` — delay apology
- `enterprise_priority_response` — enterprise urgency acknowledgment

---

## 👁 Observation Space

Each step returns a structured `Observation`:

```json
{
  "ticket": {
    "ticket_id": "TKT-003",
    "subject": "UNACCEPTABLE - Wrong item shipped...",
    "messages": [
      {"role": "customer", "content": "...", "timestamp": 1234567890.0, "is_internal": false}
    ],
    "sentiment": "angry",
    "priority": null,
    "category": null,
    "customer": {
      "customer_id": "CUST-1001",
      "name": "Theodora Vance",
      "account_tier": "enterprise",
      "account_age_days": 730,
      "total_orders": 45,
      "previous_tickets": 12,
      "is_verified": true
    },
    "order_id": "ORD-55102",
    "order_amount": 4800.00
  },
  "queue_stats": {
    "total_open": 47,
    "critical_count": 3,
    "high_count": 11,
    "avg_wait_minutes": 18.5,
    "sla_breached_count": 2
  },
  "available_macros": ["greeting_standard", "billing_refund_initiated", ...],
  "available_actions": ["triage", "internal_note"],
  "step_number": 0,
  "max_steps": 10,
  "episode_id": "f47ac10b-...",
  "task_id": "task_3_enterprise_escalation",
  "task_description": "An enterprise customer...",
  "elapsed_seconds": 0.01
}
```

---

## 💰 Reward Function

The reward is **dense** — providing a signal on every step, not just at the end.

| Component | Value | Trigger |
|-----------|-------|---------|
| `triage_accuracy` | +0.15 / -0.10 | Priority correct/wrong |
| `triage_accuracy` | +0.10 / -0.05 | Category correct/wrong |
| `response_quality` | +0.05 to +0.10 | Message word count appropriateness |
| `empathy_bonus` | +0.05 | First response to angry/frustrated customer uses empathy keywords |
| `escalation_accuracy` | +0.15 | Escalated to correct team |
| `escalation_accuracy` | -0.05 | Escalated to wrong team |
| `resolution` | +0.20 | Resolved after engaging with customer |
| `efficiency_bonus` | +0.05 | Resolved in ≤50% of max steps |
| `internal_note` | +0.01 to +0.03 | Substantive internal note added |
| `premature_close` | -0.15 | Closed without resolving (never appropriate here) |
| `repetition_penalty` | -0.08 × n | Same action repeated ≥3 times |

**Reward range:** `[-1.0, 1.0]`  
**Episode return range:** Approximately `[-0.5, 0.8]` for most agents

---

## 🚀 Setup & Usage

### Option 1: Docker (recommended)

```bash
git clone https://github.com/openenv/support-triage-v1
cd support-triage-v1

docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
```

The environment is now available at `http://localhost:7860`.

### Option 2: Local Python

```bash
pip install -r requirements.txt
python app.py
```

### Option 3: Hugging Face Spaces

The environment is deployed at: `https://huggingface.co/spaces/openenv/support-triage-v1`

---

## 🌐 HTTP API

### Start a new episode
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_billing_duplicate"}'
```

Response includes `episode_id` and initial `observation`.

### Take a step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id": "<your-episode-id>",
    "action": {
      "action_type": "triage",
      "priority": "medium",
      "category": "billing"
    }
  }'
```

### Get current state
```bash
curl http://localhost:7860/state?episode_id=<your-episode-id>
```

### Grade current episode
```bash
curl -X POST "http://localhost:7860/grade?episode_id=<your-episode-id>"
```

### Interactive docs
Visit `http://localhost:7860/docs` for full Swagger UI.

---

## 🐍 Python API (local)

```python
from env.environment import SupportTriageEnv
from env.models import Action, ActionType, Priority, Category, EscalationTeam

env = SupportTriageEnv()

# Task 1: Billing duplicate
obs = env.reset("task_1_billing_duplicate")
print(f"Ticket: {obs.ticket.subject}")
print(f"Customer: {obs.ticket.customer.name} ({obs.ticket.customer.account_tier})")

# Step 1: Triage
result = env.step(Action(
    action_type=ActionType.TRIAGE,
    priority=Priority.MEDIUM,
    category=Category.BILLING
))
print(f"Triage reward: {result.reward.value:.3f}")

# Step 2: Acknowledge
result = env.step(Action(
    action_type=ActionType.RESPOND,
    message="I sincerely apologize for the duplicate charge, Maria. I'll look into this right away."
))

# Step 3: Request info
result = env.step(Action(
    action_type=ActionType.REQUEST_INFO,
    message="Could you please share the transaction date and last 4 digits from your statement?"
))

# Step 4: Escalate
result = env.step(Action(
    action_type=ActionType.ESCALATE,
    escalation_team=EscalationTeam.REFUNDS_TEAM,
    escalation_reason="Duplicate charge confirmed, needs refund processing"
))

# Step 5: Resolve
result = env.step(Action(
    action_type=ActionType.RESOLVE,
    resolution_summary="Duplicate charge acknowledged, transaction info requested, escalated to refunds team."
))

print(f"Done: {result.done}")
print(f"Final score: {result.info.get('final_score', 0):.3f}")
print(f"Grade breakdown: {result.info.get('final_grade', {}).get('breakdown', {})}")

# Full state
state = env.state()
print(f"Total reward: {state.cumulative_reward:.3f}")
```

---

## 📊 Baseline Scores

Run against all 3 tasks using GPT-4o-mini:

```bash
export OPENAI_API_KEY=sk-...
python baseline/run_baseline.py --mode local
```

**Baseline results (gpt-4o-mini, temperature=0.2):**

| Task | Difficulty | Mean Score | Key Weakness |
|------|-----------|------------|--------------|
| task_1_billing_duplicate | Easy | **0.72** | Sometimes skips explicit info request |
| task_2_technical_crash | Medium | **0.58** | Misses workaround; partial diagnostic coverage |
| task_3_enterprise_escalation | Hard | **0.38** | Deflects with policy language; misses policy override |
| **Overall** | — | **0.56** | — |

These scores provide meaningful headroom: a well-tuned agent should score 0.85+ on Task 1, 0.75+ on Task 2, and 0.65+ on Task 3.

---

## 🧪 Running Tests

```bash
# Full pytest suite (requires pip install pytest pydantic fastapi)
python -m pytest tests/test_environment.py -v

# Standalone validation (no external deps — works anywhere)
python tests/validate_logic.py
```

---

## 🔍 OpenEnv Validation

```bash
openenv validate openenv.yaml
```

Or verify manually:
```bash
curl http://localhost:7860/health
curl http://localhost:7860/openenv.yaml
curl http://localhost:7860/tasks
```

---

## 🏗 Environment Design Notes

### State Management
Each `reset()` creates a fresh `EpisodeState` with a deep copy of the ticket, zeroed counters, and a new UUID episode ID. No state leaks between episodes.

### Action Gating
`TRIAGE` must occur before `RESOLVE` or `CLOSE`. Before triage, only `TRIAGE` and `INTERNAL_NOTE` are shown as available actions. This enforces correct workflow ordering.

### Reward Shaping Philosophy
The reward function is designed to be **informative but not manipulable**:
- Correct triage gives immediate signal (agent knows quickly if it's on track)
- Response quality rewards are proportional to word count appropriateness (not just "longer = better")
- Repetition penalty discourages prompt-stuffing loops
- Empathy bonus only fires on *first* response to an emotional customer (not on every message)

### Grader Design
Graders use keyword density analysis, flag tracking, and structural checks rather than semantic similarity — making them fast, deterministic, and exploit-resistant. The Task 3 grader specifically penalizes deflection language (common in LLM responses) which is a genuine hard problem.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

Built for the OpenEnv hackathon. Inspired by real customer support workflows at scale.
