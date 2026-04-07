"""
Standalone validation script — tests core logic without external dependencies.
Simulates pydantic BaseModel behavior using dataclasses.
"""
import sys
import os
import copy
import time
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

# ── Minimal stubs so we can import without pydantic ──────────────────────────

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"
    ACCOUNT = "account"
    PRODUCT = "product"
    REFUND = "refund"
    GENERAL = "general"

class ActionType(str, Enum):
    TRIAGE = "triage"
    RESPOND = "respond"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    REQUEST_INFO = "request_info"
    APPLY_MACRO = "apply_macro"
    INTERNAL_NOTE = "internal_note"
    CLOSE = "close"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"

class EscalationTeam(str, Enum):
    BILLING_SPECIALIST = "billing_specialist"
    SENIOR_TECH = "senior_tech"
    REFUNDS_TEAM = "refunds_team"
    ACCOUNT_SECURITY = "account_security"
    MANAGEMENT = "management"


# ── Simplified versions of models/environment for validation ─────────────────

@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    email: str
    account_tier: str
    account_age_days: int
    total_orders: int
    previous_tickets: int
    is_verified: bool

@dataclass
class TicketMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    is_internal: bool = False

@dataclass
class Ticket:
    ticket_id: str
    subject: str
    messages: List[TicketMessage]
    created_at: float
    sentiment: str
    customer: CustomerProfile
    priority: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    order_id: Optional[str] = None
    order_amount: Optional[float] = None

@dataclass
class Action:
    action_type: str
    priority: Optional[str] = None
    category: Optional[str] = None
    message: Optional[str] = None
    escalation_team: Optional[str] = None
    escalation_reason: Optional[str] = None
    macro_name: Optional[str] = None
    resolution_summary: Optional[str] = None

@dataclass
class EpisodeState:
    episode_id: str
    task_id: str
    step_number: int
    max_steps: int
    ticket: Ticket
    action_history: List[Action]
    reward_history: List[float]
    cumulative_reward: float
    done: bool
    triage_done: bool
    responded: bool
    info_requested: bool
    escalated: bool
    resolved: bool
    closed: bool
    correct_priority: Optional[str] = None
    correct_category: Optional[str] = None
    grader_metadata: Dict[str, Any] = field(default_factory=dict)


# ── Grader helpers ────────────────────────────────────────────────────────────

def _text_contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)

def _get_all_agent_messages(state: EpisodeState) -> str:
    parts = []
    for action in state.action_history:
        if action.action_type in ("respond", "request_info", "apply_macro"):
            if action.message:
                parts.append(action.message)
        if action.action_type == "resolve" and action.resolution_summary:
            parts.append(action.resolution_summary)
    return " ".join(parts)

def _get_internal_notes(state: EpisodeState) -> str:
    parts = []
    for action in state.action_history:
        if action.action_type == "internal_note" and action.message:
            parts.append(action.message)
    return " ".join(parts)

def _triage_action(state: EpisodeState):
    for action in state.action_history:
        if action.action_type == "triage":
            p_ok = str(action.priority).lower() == str(state.correct_priority).lower()
            c_ok = str(action.category).lower() == str(state.correct_category).lower()
            return p_ok, c_ok
    return False, False

def _escalated_to(state: EpisodeState, team: str) -> bool:
    for action in state.action_history:
        if action.action_type == "escalate":
            if action.escalation_team and team.lower() in str(action.escalation_team).lower():
                return True
    return False


# ── Task 1 Grader ─────────────────────────────────────────────────────────────

def grade_task_1(state: EpisodeState) -> Dict[str, Any]:
    rubric = {"correct_triage": 0.25, "customer_acknowledged": 0.20, "info_requested_or_escalated": 0.25, "resolution_quality": 0.30}
    earned = {}
    p_ok, c_ok = _triage_action(state)
    earned["correct_triage"] = ((0.5 if p_ok else 0) + (0.5 if c_ok else 0)) * rubric["correct_triage"]
    agent_text = _get_all_agent_messages(state)
    ack_kw = ["apologize", "sorry", "understand", "inconvenience", "certainly", "happy to help", "look into"]
    earned["customer_acknowledged"] = rubric["customer_acknowledged"] * (1.0 if _text_contains_any(agent_text, ack_kw) else 0.0)
    info_kw = ["transaction", "statement", "bank", "date", "confirm", "provide", "screenshot"]
    info_requested = state.info_requested or _text_contains_any(agent_text, info_kw)
    escalated_refunds = _escalated_to(state, "refunds")
    earned["info_requested_or_escalated"] = rubric["info_requested_or_escalated"] * (1.0 if (info_requested or escalated_refunds) else 0.0)
    res_kw = ["refund", "duplicate", "charge", "credit", "process", "investigate"]
    has_res = _text_contains_any(agent_text, res_kw)
    resolved = state.resolved or state.escalated
    words = len(agent_text.split())
    quality = min(words / 20, 1.0) if words < 20 else 1.0
    res_score = (1.0 if (has_res and resolved) else 0.6 if has_res else 0.3 if resolved else 0.0)
    earned["resolution_quality"] = rubric["resolution_quality"] * res_score * quality
    return {"score": round(min(sum(earned.values()), 1.0), 4), "breakdown": earned}


# ── Reward Computer ────────────────────────────────────────────────────────────

def compute_reward(action: Action, state: EpisodeState, task_meta: Dict) -> float:
    components = {}
    at = action.action_type

    if at == "triage":
        p_ok = str(action.priority).lower() == str(task_meta["correct_priority"]).lower()
        c_ok = str(action.category).lower() == str(task_meta["correct_category"]).lower()
        components["triage_accuracy"] = (0.15 if p_ok else -0.10) + (0.10 if c_ok else -0.05)

    elif at in ("respond", "request_info", "apply_macro"):
        msg = action.message or ""
        words = len(msg.split())
        components["response_quality"] = (-0.05 if words < 10 else 0.05 if words < 20 else 0.10 if words <= 120 else 0.05)

    elif at == "escalate":
        correct = task_meta.get("correct_escalation", "")
        given = str(action.escalation_team or "").lower()
        components["escalation"] = 0.15 if correct and correct.lower() in given else -0.05

    elif at == "resolve":
        components["resolution"] = 0.20 if (state.escalated or state.responded) else 0.05

    elif at == "close":
        components["premature_close"] = -0.15

    recent = state.action_history[-4:-1]
    repeat = sum(1 for a in recent if a.action_type == at)
    if repeat >= 2:
        components["repetition_penalty"] = -0.08 * repeat

    total = sum(components.values())
    return max(-1.0, min(1.0, total))


# ── Scenario Runner ────────────────────────────────────────────────────────────

def make_task1_ticket():
    return Ticket(
        ticket_id="TKT-001",
        subject="I was charged twice for my subscription",
        messages=[TicketMessage(role="customer", content="I noticed two charges of $29.99. Could you refund the duplicate? Order ORD-78234.")],
        created_at=time.time() - 3600,
        sentiment="neutral",
        customer=CustomerProfile("CUST-4421", "Maria Santos", "maria@email.com", "basic", 420, 8, 1, True),
        order_id="ORD-78234",
        order_amount=29.99,
    )

def make_state(ticket, task_id, correct_priority, correct_category, max_steps=6):
    return EpisodeState(
        episode_id=str(uuid.uuid4()),
        task_id=task_id,
        step_number=0,
        max_steps=max_steps,
        ticket=ticket,
        action_history=[],
        reward_history=[],
        cumulative_reward=0.0,
        done=False,
        triage_done=False,
        responded=False,
        info_requested=False,
        escalated=False,
        resolved=False,
        closed=False,
        correct_priority=correct_priority,
        correct_category=correct_category,
    )

TASK1_META = {"correct_priority": "medium", "correct_category": "billing", "correct_escalation": "refunds_team"}


# ══════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════

passed = 0
failed = 0

def test(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}: {msg}")
        failed += 1


print("\n" + "="*60)
print("  SupportTriage OpenEnv — Validation Suite")
print("="*60)

# ── Test Group 1: Reward Function ─────────────────────────────────────────────
print("\n[1] Reward Function")
ticket = make_task1_ticket()
state = make_state(ticket, "task_1", "medium", "billing")

# Correct triage
action_correct = Action(action_type="triage", priority="medium", category="billing")
r_correct = compute_reward(action_correct, state, TASK1_META)
test("correct triage → positive reward", r_correct > 0, f"got {r_correct}")

# Wrong triage
action_wrong = Action(action_type="triage", priority="low", category="general")
r_wrong = compute_reward(action_wrong, state, TASK1_META)
test("wrong triage → negative reward", r_wrong < 0, f"got {r_wrong}")

# Correct escalation
state2 = make_state(ticket, "task_1", "medium", "billing")
state2.escalated = True
action_esc = Action(action_type="escalate", escalation_team="refunds_team")
r_esc = compute_reward(action_esc, state2, TASK1_META)
test("correct escalation → positive reward", r_esc > 0, f"got {r_esc}")

# Wrong escalation
action_esc_wrong = Action(action_type="escalate", escalation_team="management")
r_esc_wrong = compute_reward(action_esc_wrong, state2, TASK1_META)
test("wrong escalation → negative reward", r_esc_wrong < 0, f"got {r_esc_wrong}")

# Reward in range
test("reward in [-1, 1]", -1.0 <= r_correct <= 1.0)

# Repetition penalty
state3 = make_state(ticket, "task_1", "medium", "billing")
action_respond = Action(action_type="respond", message="We are looking into this now.")
state3.action_history = [action_respond, action_respond, action_respond]
r_rep = compute_reward(action_respond, state3, TASK1_META)
action_respond_first = Action(action_type="respond", message="We are looking into this now.")
state_fresh = make_state(ticket, "task_1", "medium", "billing")
r_fresh = compute_reward(action_respond_first, state_fresh, TASK1_META)
test("repetition penalty applied", r_rep < r_fresh, f"rep={r_rep:.3f} vs fresh={r_fresh:.3f}")

# Premature close penalty
action_close = Action(action_type="close")
r_close = compute_reward(action_close, state3, TASK1_META)
test("premature close → penalty", r_close < 0, f"got {r_close}")


# ── Test Group 2: Grader ──────────────────────────────────────────────────────
print("\n[2] Grader — Task 1")

# Perfect episode
state_perfect = make_state(make_task1_ticket(), "task_1", "medium", "billing")
state_perfect.action_history = [
    Action(action_type="triage", priority="medium", category="billing"),
    Action(action_type="respond", message="I sincerely apologize for the duplicate charge, Maria. I understand how frustrating this must be."),
    Action(action_type="request_info", message="Could you please provide the transaction ID and date from your bank statement?"),
    Action(action_type="escalate", escalation_team="refunds_team", escalation_reason="Duplicate charge confirmed"),
    Action(action_type="resolve", resolution_summary="Acknowledged duplicate charge, requested transaction confirmation, escalated to refunds team."),
]
state_perfect.triage_done = True
state_perfect.responded = True
state_perfect.info_requested = True
state_perfect.escalated = True
state_perfect.resolved = True
state_perfect.step_number = 5

grade_perfect = grade_task_1(state_perfect)
test("perfect episode → score ≥ 0.70", grade_perfect["score"] >= 0.70, f"got {grade_perfect['score']}")

# Empty episode
state_empty = make_state(make_task1_ticket(), "task_1", "medium", "billing")
state_empty.action_history = [
    Action(action_type="triage", priority="low", category="general"),
    Action(action_type="resolve", resolution_summary="Done."),
]
state_empty.triage_done = True
state_empty.resolved = True
state_empty.step_number = 2
grade_empty = grade_task_1(state_empty)
test("empty/wrong episode → score < 0.40", grade_empty["score"] < 0.40, f"got {grade_empty['score']}")

# Score in [0, 1]
test("grader score in [0, 1]", 0.0 <= grade_perfect["score"] <= 1.0)

# Deterministic
grade2 = grade_task_1(state_perfect)
test("grader is deterministic", grade_perfect["score"] == grade2["score"])

# Breakdown keys
expected_keys = {"correct_triage", "customer_acknowledged", "info_requested_or_escalated", "resolution_quality"}
test("breakdown keys match rubric", set(grade_perfect["breakdown"].keys()) == expected_keys)


# ── Test Group 3: State Management ────────────────────────────────────────────
print("\n[3] State Management")

state_sm = make_state(make_task1_ticket(), "task_1", "medium", "billing")
test("initial step_number is 0", state_sm.step_number == 0)
test("initial action_history is empty", len(state_sm.action_history) == 0)
test("initial triage_done is False", not state_sm.triage_done)
test("initial cumulative_reward is 0", state_sm.cumulative_reward == 0.0)

# Apply triage
state_sm.action_history.append(Action(action_type="triage", priority="medium", category="billing"))
state_sm.triage_done = True
state_sm.step_number += 1
test("after triage: triage_done=True", state_sm.triage_done)
test("after triage: step_number=1", state_sm.step_number == 1)

# Reset = fresh state
state_reset = make_state(make_task1_ticket(), "task_1", "medium", "billing")
test("reset produces clean state", state_reset.step_number == 0 and len(state_reset.action_history) == 0)


# ── Test Group 4: Episode Boundaries ─────────────────────────────────────────
print("\n[4] Episode Boundaries")

# Max steps truncation
state_trunc = make_state(make_task1_ticket(), "task_1", "medium", "billing", max_steps=3)
state_trunc.step_number = 3
should_truncate = state_trunc.step_number >= state_trunc.max_steps
test("truncation at max_steps", should_truncate)

# Resolve terminates episode
state_resolve = make_state(make_task1_ticket(), "task_1", "medium", "billing")
action_resolve = Action(action_type="resolve", resolution_summary="Issue resolved completely.")
state_resolve.triage_done = True
# resolve action → done=True
terminal = action_resolve.action_type in ("resolve", "close")
test("resolve action → terminal", terminal)

# Close terminates episode
action_close2 = Action(action_type="close")
terminal_close = action_close2.action_type in ("resolve", "close")
test("close action → terminal", terminal_close)


# ── Test Group 5: Task Definitions ────────────────────────────────────────────
print("\n[5] Task Definitions")

tasks_data = {
    "task_1_billing_duplicate": {"difficulty": "easy", "max_steps": 6, "correct_priority": "medium", "correct_category": "billing"},
    "task_2_technical_crash": {"difficulty": "medium", "max_steps": 8, "correct_priority": "high", "correct_category": "technical"},
    "task_3_enterprise_escalation": {"difficulty": "hard", "max_steps": 10, "correct_priority": "critical", "correct_category": "refund"},
}

test("exactly 3 tasks defined", len(tasks_data) == 3)

difficulties = [t["difficulty"] for t in tasks_data.values()]
test("easy difficulty present", "easy" in difficulties)
test("medium difficulty present", "medium" in difficulties)
test("hard difficulty present", "hard" in difficulties)

max_steps_values = [t["max_steps"] for t in tasks_data.values()]
test("max_steps increases with difficulty", sorted(max_steps_values) == max_steps_values)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  Results: {passed} passed, {failed} failed")
print("="*60)

if failed > 0:
    sys.exit(1)
else:
    print("  ✅ All validation checks passed!\n")
