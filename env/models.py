"""
OpenEnv: Customer Support Triage & Resolution Environment
Typed Pydantic models for Observation, Action, and Reward.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

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


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class ActionType(str, Enum):
    TRIAGE = "triage"          # Assign priority + category
    RESPOND = "respond"        # Send a reply to customer
    ESCALATE = "escalate"      # Escalate to human agent / specialist
    RESOLVE = "resolve"        # Mark ticket as resolved
    REQUEST_INFO = "request_info"  # Ask customer for more information
    APPLY_MACRO = "apply_macro"    # Apply a pre-written template/macro
    INTERNAL_NOTE = "internal_note"  # Add internal note (not visible to customer)
    CLOSE = "close"            # Close without resolving (spam, duplicate)


class EscalationTeam(str, Enum):
    BILLING_SPECIALIST = "billing_specialist"
    SENIOR_TECH = "senior_tech"
    REFUNDS_TEAM = "refunds_team"
    ACCOUNT_SECURITY = "account_security"
    MANAGEMENT = "management"


# ─────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────

class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    email: str
    account_tier: str = Field(..., description="free | basic | premium | enterprise")
    account_age_days: int
    total_orders: int
    previous_tickets: int
    is_verified: bool


class TicketMessage(BaseModel):
    role: str = Field(..., description="customer | agent | system")
    content: str
    timestamp: float = Field(default_factory=time.time)
    is_internal: bool = False


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    messages: List[TicketMessage]
    created_at: float
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    sentiment: Sentiment
    customer: CustomerProfile
    tags: List[str] = Field(default_factory=list)
    order_id: Optional[str] = None
    order_amount: Optional[float] = None


class QueueStats(BaseModel):
    total_open: int
    critical_count: int
    high_count: int
    avg_wait_minutes: float
    sla_breached_count: int


# ─────────────────────────────────────────────
# Core OpenEnv Types
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket: Ticket
    queue_stats: QueueStats
    available_macros: List[str] = Field(
        default_factory=list,
        description="List of available response template names"
    )
    available_actions: List[ActionType]
    step_number: int
    max_steps: int
    episode_id: str
    task_id: str
    task_description: str
    elapsed_seconds: float = 0.0

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """An action the agent can take."""
    action_type: ActionType
    # For TRIAGE
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    # For RESPOND / REQUEST_INFO / INTERNAL_NOTE
    message: Optional[str] = None
    # For ESCALATE
    escalation_team: Optional[EscalationTeam] = None
    escalation_reason: Optional[str] = None
    # For APPLY_MACRO
    macro_name: Optional[str] = None
    macro_variables: Optional[Dict[str, str]] = None
    # For RESOLVE / CLOSE
    resolution_summary: Optional[str] = None

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    """Reward signal returned with each step."""
    value: float = Field(..., ge=-1.0, le=1.0, description="Reward in [-1, 1]")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Named sub-rewards for interpretability"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic information"
    )


class StepResult(BaseModel):
    """Result of a step() call."""
    observation: Observation
    reward: Reward
    done: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    """Full internal state, returned by state()."""
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
    correct_priority: Optional[Priority] = None
    correct_category: Optional[Category] = None
    expected_resolution: Optional[str] = None
    grader_metadata: Dict[str, Any] = Field(default_factory=dict)
