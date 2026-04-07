"""
SupportTriageEnv — Main environment class.
Implements the OpenEnv interface: reset(), step(), state().
"""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, ActionType, Category, EpisodeState, EscalationTeam,
    Observation, Priority, QueueStats, Reward, StepResult, Ticket,
)
from tasks.task_definitions import get_task, TASKS
from graders.task_graders import grade


# Available macro names (pre-written response templates)
AVAILABLE_MACROS = [
    "greeting_standard",
    "billing_refund_initiated",
    "technical_troubleshooting_steps",
    "request_order_number",
    "request_diagnostic_info",
    "escalation_acknowledgment",
    "resolution_confirmed",
    "apology_delay",
    "enterprise_priority_response",
]

# Macro templates (expanded when applied)
MACRO_TEMPLATES = {
    "greeting_standard": "Thank you for reaching out to us! I'm here to help you today.",
    "billing_refund_initiated": (
        "I've reviewed your account and initiated a refund for the duplicate charge. "
        "You should see the credit in 3-5 business days."
    ),
    "technical_troubleshooting_steps": (
        "Let's work through this together. Could you please provide: "
        "1) Your operating system and version, 2) The app version you're using, "
        "3) Steps to reproduce the issue."
    ),
    "request_order_number": "Could you please provide your order number so I can look into this right away?",
    "request_diagnostic_info": (
        "To help diagnose this issue, could you share: your OS, app version, "
        "and the size/type of file you were trying to export?"
    ),
    "escalation_acknowledgment": (
        "I've escalated your case to our specialist team who will review it as a priority. "
        "You can expect to hear back within 2 business hours."
    ),
    "resolution_confirmed": "Great news! Your issue has been resolved. Is there anything else I can help you with?",
    "apology_delay": (
        "I sincerely apologize for the delay in resolving your issue. "
        "This is not the standard of service we aim to provide."
    ),
    "enterprise_priority_response": (
        "As an Enterprise customer, your case is being flagged for immediate priority handling. "
        "A senior account manager will be in touch within 1 hour."
    ),
}


class SupportTriageEnv:
    """
    Customer Support Triage & Resolution Environment.

    An agent must handle realistic support tickets by:
    - Triaging priority and category correctly
    - Gathering missing information
    - Crafting empathetic, helpful responses
    - Escalating appropriately
    - Resolving tickets efficiently

    OpenEnv Interface:
        reset(task_id) -> Observation
        step(action)   -> StepResult(observation, reward, done, truncated, info)
        state()        -> EpisodeState
    """

    VERSION = "1.0.0"
    ENVIRONMENT_ID = "support-triage-v1"

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self._state: Optional[EpisodeState] = None
        self._task_meta: Optional[Dict[str, Any]] = None
        self._episode_start_time: float = 0.0

    # ─────────────────────────────────────────
    # OpenEnv Core Methods
    # ─────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment for a new episode.
        Returns the initial Observation.
        """
        if task_id:
            self.task_id = task_id
        if not self.task_id:
            raise ValueError("task_id must be provided to reset()")

        task = get_task(self.task_id)
        self._task_meta = task["meta"]
        ticket: Ticket = copy.deepcopy(task["ticket"])
        self._episode_start_time = time.time()

        self._state = EpisodeState(
            episode_id=str(uuid.uuid4()),
            task_id=self.task_id,
            step_number=0,
            max_steps=self._task_meta["max_steps"],
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
            correct_priority=self._task_meta.get("correct_priority"),
            correct_category=self._task_meta.get("correct_category"),
            expected_resolution=self._task_meta.get("expected_resolution"),
            grader_metadata={},
        )

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply an action. Returns (observation, reward, done, truncated, info).
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Validate and apply action
        validation_error = self._validate_action(action)
        if validation_error:
            # Invalid action: small penalty, no state change
            reward = Reward(
                value=-0.05,
                components={"invalid_action": -0.05},
                info={"error": validation_error},
            )
            obs = self._build_observation()
            return StepResult(
                observation=obs,
                reward=reward,
                done=False,
                truncated=False,
                info={"error": validation_error},
            )

        # Apply action to state
        self._state.action_history.append(action)
        self._state.step_number += 1

        # Update state flags
        self._apply_action_to_state(action)

        # Compute reward
        reward = self._compute_reward(action)
        self._state.reward_history.append(reward.value)
        self._state.cumulative_reward += reward.value

        # Check terminal conditions
        done = self._check_done(action)
        truncated = self._state.step_number >= self._state.max_steps and not done
        self._state.done = done or truncated

        obs = self._build_observation()

        info: Dict[str, Any] = {
            "step": self._state.step_number,
            "cumulative_reward": self._state.cumulative_reward,
        }

        # Run final grader if episode is complete
        if self._state.done:
            grade_result = grade(self.task_id, self._state)
            self._state.grader_metadata = grade_result
            info["final_grade"] = grade_result
            info["final_score"] = grade_result["score"]

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def state(self) -> EpisodeState:
        """Return the full current state (for inspection / debugging)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ─────────────────────────────────────────
    # Internal: Observation Builder
    # ─────────────────────────────────────────

    def _build_observation(self) -> Observation:
        elapsed = time.time() - self._episode_start_time
        return Observation(
            ticket=self._state.ticket,
            queue_stats=QueueStats(
                total_open=47,
                critical_count=3,
                high_count=11,
                avg_wait_minutes=18.5,
                sla_breached_count=2,
            ),
            available_macros=AVAILABLE_MACROS,
            available_actions=self._get_available_actions(),
            step_number=self._state.step_number,
            max_steps=self._state.max_steps,
            episode_id=self._state.episode_id,
            task_id=self._state.task_id,
            task_description=self._task_meta["description"],
            elapsed_seconds=round(elapsed, 2),
        )

    def _get_available_actions(self) -> List[ActionType]:
        """Available actions depend on current state."""
        actions = []
        if not self._state.triage_done:
            actions.append(ActionType.TRIAGE)
        if self._state.triage_done:
            actions.extend([
                ActionType.RESPOND,
                ActionType.REQUEST_INFO,
                ActionType.APPLY_MACRO,
                ActionType.INTERNAL_NOTE,
                ActionType.ESCALATE,
                ActionType.RESOLVE,
                ActionType.CLOSE,
            ])
        else:
            # Can triage-then-immediately-do-things in same step
            actions.extend([ActionType.INTERNAL_NOTE])
        return actions

    # ─────────────────────────────────────────
    # Internal: Action Validation
    # ─────────────────────────────────────────

    def _validate_action(self, action: Action) -> Optional[str]:
        """Returns error string if action is invalid, else None."""
        at = action.action_type

        if at == ActionType.TRIAGE:
            if not action.priority:
                return "TRIAGE requires 'priority'"
            if not action.category:
                return "TRIAGE requires 'category'"

        elif at in (ActionType.RESPOND, ActionType.REQUEST_INFO, ActionType.INTERNAL_NOTE):
            if not action.message or len(action.message.strip()) < 5:
                return f"{at.upper()} requires a non-empty 'message' (min 5 chars)"

        elif at == ActionType.ESCALATE:
            if not action.escalation_team:
                return "ESCALATE requires 'escalation_team'"

        elif at == ActionType.APPLY_MACRO:
            if not action.macro_name:
                return "APPLY_MACRO requires 'macro_name'"
            if action.macro_name not in MACRO_TEMPLATES:
                return f"Unknown macro: {action.macro_name}. Available: {list(MACRO_TEMPLATES.keys())}"

        elif at == ActionType.RESOLVE:
            if not self._state.triage_done:
                return "Cannot RESOLVE before triaging (TRIAGE action required first)"
            if not action.resolution_summary or len(action.resolution_summary.strip()) < 10:
                return "RESOLVE requires a 'resolution_summary' (min 10 chars)"

        elif at == ActionType.CLOSE:
            if not self._state.triage_done:
                return "Cannot CLOSE before triaging"

        return None

    # ─────────────────────────────────────────
    # Internal: State Updates
    # ─────────────────────────────────────────

    def _apply_action_to_state(self, action: Action) -> None:
        at = action.action_type
        if at == ActionType.TRIAGE:
            self._state.triage_done = True
            self._state.ticket.priority = action.priority
            self._state.ticket.category = action.category
        elif at == ActionType.RESPOND:
            self._state.responded = True
        elif at == ActionType.REQUEST_INFO:
            self._state.info_requested = True
        elif at == ActionType.ESCALATE:
            self._state.escalated = True
        elif at == ActionType.RESOLVE:
            self._state.resolved = True
        elif at == ActionType.CLOSE:
            self._state.closed = True
        elif at == ActionType.APPLY_MACRO:
            # Macro counts as a response
            self._state.responded = True
            # Attach expanded content to action
            action.message = MACRO_TEMPLATES.get(action.macro_name, "")

    # ─────────────────────────────────────────
    # Internal: Reward Shaping
    # ─────────────────────────────────────────

    def _compute_reward(self, action: Action) -> Reward:
        """
        Dense reward function — provides signal throughout the episode.
        Components:
          - Triage accuracy (immediate, one-time)
          - Response quality (content-based)
          - Escalation correctness
          - Efficiency bonus (fewer steps = better)
          - Waste penalties (loops, empty actions)
        """
        components: Dict[str, float] = {}
        info: Dict[str, Any] = {}
        at = action.action_type

        # ── Triage reward (one-time) ──
        if at == ActionType.TRIAGE:
            p_ok = str(action.priority).lower() == str(self._state.correct_priority).lower()
            c_ok = str(action.category).lower() == str(self._state.correct_category).lower()
            triage_reward = (0.15 if p_ok else -0.10) + (0.10 if c_ok else -0.05)
            components["triage_accuracy"] = triage_reward
            info["priority_correct"] = p_ok
            info["category_correct"] = c_ok

        # ── Response quality (for RESPOND / REQUEST_INFO / APPLY_MACRO) ──
        elif at in (ActionType.RESPOND, ActionType.REQUEST_INFO, ActionType.APPLY_MACRO):
            msg = action.message or ""
            words = len(msg.split())
            # Too short: penalty, good length: reward
            if words < 10:
                components["response_quality"] = -0.05
            elif words < 20:
                components["response_quality"] = 0.05
            elif words <= 120:
                components["response_quality"] = 0.10
            else:
                components["response_quality"] = 0.05  # Too verbose
            info["message_words"] = words

            # Bonus: first response (acknowledges urgency)
            if not self._state.responded and self._state.ticket.sentiment in ("angry", "frustrated"):
                empathy_kw = ["sorry", "apologize", "understand", "urgency", "priority"]
                if any(kw in msg.lower() for kw in empathy_kw):
                    components["empathy_bonus"] = 0.05

        # ── Escalation reward ──
        elif at == ActionType.ESCALATE:
            correct_team = self._task_meta.get("correct_escalation", "")
            given_team = str(action.escalation_team or "").lower()
            if correct_team and correct_team.lower() in given_team:
                components["escalation_accuracy"] = 0.15
            else:
                components["escalation_accuracy"] = -0.05  # Wrong team
            info["escalation_team"] = given_team

        # ── Resolve reward ──
        elif at == ActionType.RESOLVE:
            if self._state.escalated or self._state.responded:
                components["resolution"] = 0.20
            else:
                components["resolution"] = 0.05  # Resolved without engaging
            # Efficiency bonus
            steps_used = self._state.step_number
            max_steps = self._state.max_steps
            if steps_used <= max_steps * 0.5:
                components["efficiency_bonus"] = 0.05

        # ── Close penalty (only appropriate for spam/duplicates) ──
        elif at == ActionType.CLOSE:
            # Tickets in this env are never spam → penalize premature close
            components["premature_close"] = -0.15

        # ── Internal note: small reward for capturing context ──
        elif at == ActionType.INTERNAL_NOTE:
            msg = action.message or ""
            if len(msg.split()) >= 15:
                components["internal_note"] = 0.03
            else:
                components["internal_note"] = 0.01

        # ── Loop / repetition penalty ──
        recent = self._state.action_history[-4:-1]
        repeat_count = sum(1 for a in recent if a.action_type == at)
        if repeat_count >= 2:
            components["repetition_penalty"] = -0.08 * repeat_count

        total = sum(components.values())
        total = max(-1.0, min(1.0, total))

        return Reward(value=round(total, 4), components=components, info=info)

    # ─────────────────────────────────────────
    # Internal: Terminal Condition
    # ─────────────────────────────────────────

    def _check_done(self, action: Action) -> bool:
        at = action.action_type
        if at in (ActionType.RESOLVE, ActionType.CLOSE):
            return True
        if self._state.step_number >= self._state.max_steps:
            return True
        return False

    # ─────────────────────────────────────────
    # Convenience: run full episode from action list
    # ─────────────────────────────────────────

    def run_episode(self, actions: List[Action]) -> Dict[str, Any]:
        """Run a full episode with a list of actions. Returns summary."""
        obs = self.reset()
        results = []
        for action in actions:
            result = self.step(action)
            results.append(result)
            if result.done or result.truncated:
                break
        final = results[-1] if results else None
        return {
            "episode_id": self._state.episode_id,
            "task_id": self.task_id,
            "steps": self._state.step_number,
            "cumulative_reward": self._state.cumulative_reward,
            "done": self._state.done,
            "final_score": (final.info.get("final_score", 0.0) if final else 0.0),
            "grade_breakdown": (final.info.get("final_grade", {}) if final else {}),
        }
