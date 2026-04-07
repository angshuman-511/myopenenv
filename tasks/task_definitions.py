"""
Task definitions for the Customer Support Triage environment.
Each task defines a scenario, correct answers, and grading criteria.

TASK DIFFICULTY:
  Task 1 (easy):   Simple billing inquiry — clear intent, polite customer, obvious solution
  Task 2 (medium): Technical issue with incomplete info — requires diagnosis and info gathering
  Task 3 (hard):   Escalating angry customer with policy edge case + SLA pressure
"""

from __future__ import annotations
import time
from typing import Dict, Any
from env.models import (
    Ticket, TicketMessage, CustomerProfile, Category, Priority,
    Sentiment, EpisodeState
)


# ─────────────────────────────────────────────
# TASK 1 — EASY: Overcharge / Billing Inquiry
# ─────────────────────────────────────────────

TASK_1_TICKET = Ticket(
    ticket_id="TKT-001",
    subject="I was charged twice for my subscription",
    messages=[
        TicketMessage(
            role="customer",
            content=(
                "Hi, I noticed two charges of $29.99 on my credit card this month "
                "for my Basic subscription. Could you please refund the duplicate charge? "
                "My order number is ORD-78234. Thank you!"
            ),
            timestamp=time.time() - 3600,
        )
    ],
    created_at=time.time() - 3600,
    sentiment=Sentiment.NEUTRAL,
    customer=CustomerProfile(
        customer_id="CUST-4421",
        name="Maria Santos",
        email="maria.santos@email.com",
        account_tier="basic",
        account_age_days=420,
        total_orders=8,
        previous_tickets=1,
        is_verified=True,
    ),
    tags=[],
    order_id="ORD-78234",
    order_amount=29.99,
)

TASK_1_META: Dict[str, Any] = {
    "task_id": "task_1_billing_duplicate",
    "description": (
        "A verified Basic-tier customer reports a duplicate charge of $29.99. "
        "Triage correctly (billing, medium priority), acknowledge the issue, "
        "request confirmation of the duplicate transaction, and route to the "
        "refunds team if needed. The customer is polite and provides an order number."
    ),
    "correct_priority": Priority.MEDIUM,
    "correct_category": Category.BILLING,
    "correct_escalation": "refunds_team",
    "requires_info_request": True,  # Agent should ask for transaction IDs before refunding
    "resolution_keywords": ["refund", "duplicate", "charge", "apologize", "sorry"],
    "max_steps": 6,
    "difficulty": "easy",
    "scoring_rubric": {
        "correct_triage": 0.25,
        "customer_acknowledged": 0.20,
        "info_requested_or_escalated": 0.25,
        "resolution_quality": 0.30,
    },
}


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM: Technical Issue (Incomplete Info)
# ─────────────────────────────────────────────

TASK_2_TICKET = Ticket(
    ticket_id="TKT-002",
    subject="App keeps crashing when I try to export",
    messages=[
        TicketMessage(
            role="customer",
            content=(
                "Your app is broken. Every time I click 'Export to PDF' it just freezes "
                "and then crashes. I need this for a presentation tomorrow morning. "
                "Please fix ASAP."
            ),
            timestamp=time.time() - 7200,
        ),
        TicketMessage(
            role="customer",
            content=(
                "I tried restarting and it's still happening. This is really frustrating."
            ),
            timestamp=time.time() - 6800,
        ),
    ],
    created_at=time.time() - 7200,
    sentiment=Sentiment.FRUSTRATED,
    customer=CustomerProfile(
        customer_id="CUST-9983",
        name="James Whitfield",
        email="j.whitfield@corp.io",
        account_tier="premium",
        account_age_days=185,
        total_orders=22,
        previous_tickets=4,
        is_verified=True,
    ),
    tags=["export", "crash", "pdf"],
    order_id=None,
    order_amount=None,
)

TASK_2_META: Dict[str, Any] = {
    "task_id": "task_2_technical_crash",
    "description": (
        "A frustrated Premium customer's app crashes on PDF export. "
        "Critical info is missing: OS, app version, file size. "
        "The agent must triage correctly (technical, high priority due to premium tier), "
        "empathize with the urgency, request the missing diagnostic info "
        "(OS/version/file details), and optionally provide a workaround while escalating "
        "to senior tech support."
    ),
    "correct_priority": Priority.HIGH,
    "correct_category": Category.TECHNICAL,
    "correct_escalation": "senior_tech",
    "required_info_to_request": ["os", "version", "operating system", "app version", "file"],
    "workaround_keywords": ["workaround", "alternative", "meanwhile", "try", "download"],
    "max_steps": 8,
    "difficulty": "medium",
    "scoring_rubric": {
        "correct_triage": 0.20,
        "empathy_for_urgency": 0.15,
        "diagnostic_info_requested": 0.25,
        "workaround_offered": 0.15,
        "escalation_to_correct_team": 0.25,
    },
}


# ─────────────────────────────────────────────
# TASK 3 — HARD: Angry Enterprise Customer, Policy Edge Case, SLA Breach
# ─────────────────────────────────────────────

TASK_3_TICKET = Ticket(
    ticket_id="TKT-003",
    subject="UNACCEPTABLE - Wrong item shipped, 3 week delay, need FULL refund NOW",
    messages=[
        TicketMessage(
            role="customer",
            content=(
                "I placed order ORD-55102 for an enterprise license of 50 seats "
                "($4,800) THREE WEEKS AGO. The wrong product was delivered and despite "
                "calling 4 times, nothing has been done. My team can't work. "
                "I want a FULL refund AND compensation immediately or I'm going to "
                "my bank and social media. This is absolutely ridiculous."
            ),
            timestamp=time.time() - 1800,
        ),
        TicketMessage(
            role="customer",
            content=(
                "I just spoke with someone named 'Alex' who promised a callback within "
                "2 hours. That was yesterday. Your company is a complete joke. "
                "My legal team is on standby."
            ),
            timestamp=time.time() - 900,
        ),
    ],
    created_at=time.time() - 1800,
    sentiment=Sentiment.ANGRY,
    customer=CustomerProfile(
        customer_id="CUST-1001",
        name="Theodora Vance",
        email="tvance@megacorp-llc.com",
        account_tier="enterprise",
        account_age_days=730,
        total_orders=45,
        previous_tickets=12,
        is_verified=True,
    ),
    tags=["wrong-item", "refund", "enterprise", "SLA-breach", "legal-threat", "escalation"],
    order_id="ORD-55102",
    order_amount=4800.00,
)

TASK_3_META: Dict[str, Any] = {
    "task_id": "task_3_enterprise_escalation",
    "description": (
        "An enterprise customer ($4,800 order) is furious: wrong product delivered, "
        "3-week delay, 4 failed contacts, and now threatening legal action and chargeback. "
        "The agent must: triage as CRITICAL, acknowledge past failures without deflecting, "
        "de-escalate the anger, escalate immediately to management AND refunds team, "
        "offer concrete next steps with a specific callback commitment, "
        "and add an internal note with full context. "
        "Policy edge case: refunds >$500 normally require manager approval, "
        "but SLA breach + enterprise tier overrides standard policy. "
        "Agent must recognize this override."
    ),
    "correct_priority": Priority.CRITICAL,
    "correct_category": Category.REFUND,
    "correct_escalation": "management",
    "secondary_escalation": "refunds_team",
    "requires_internal_note": True,
    "de_escalation_keywords": [
        "sincerely apologize", "unacceptable", "failed", "understand your frustration",
        "take ownership", "priority", "personally"
    ],
    "commitment_keywords": ["within", "hours", "callback", "contact", "today", "personally"],
    "policy_override_keywords": ["enterprise", "sla", "approve", "override", "exception", "authorize"],
    "internal_note_required": True,
    "max_steps": 10,
    "difficulty": "hard",
    "scoring_rubric": {
        "correct_triage_critical": 0.15,
        "acknowledgment_without_deflection": 0.20,
        "de_escalation_quality": 0.20,
        "escalation_to_management": 0.15,
        "concrete_commitment": 0.15,
        "internal_note_added": 0.10,
        "policy_override_recognized": 0.05,
    },
}


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

TASKS = {
    "task_1_billing_duplicate": {
        "ticket": TASK_1_TICKET,
        "meta": TASK_1_META,
    },
    "task_2_technical_crash": {
        "ticket": TASK_2_TICKET,
        "meta": TASK_2_META,
    },
    "task_3_enterprise_escalation": {
        "ticket": TASK_3_TICKET,
        "meta": TASK_3_META,
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks():
    return [
        {
            "task_id": t,
            "difficulty": TASKS[t]["meta"]["difficulty"],
            "description": TASKS[t]["meta"]["description"][:100] + "...",
        }
        for t in TASKS
    ]
