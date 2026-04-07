"""
Agent Graders for each task.
Each grader receives the final EpisodeState and returns a score in [0.0, 1.0]
with a detailed breakdown.

Graders are DETERMINISTIC: same EpisodeState → same score, always.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple
from env.models import (
    EpisodeState, Action, ActionType, Priority, Category, EscalationTeam
)


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _text_contains_any(text: str, keywords: List[str]) -> bool:
    """Case-insensitive keyword search."""
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _get_all_agent_messages(state: EpisodeState) -> str:
    """Collect all outbound agent message text."""
    parts = []
    for action in state.action_history:
        if action.action_type in (ActionType.RESPOND, ActionType.REQUEST_INFO, ActionType.APPLY_MACRO):
            if action.message:
                parts.append(action.message)
        if action.action_type == ActionType.RESOLVE and action.resolution_summary:
            parts.append(action.resolution_summary)
    return " ".join(parts)


def _get_internal_notes(state: EpisodeState) -> str:
    """Collect internal note text."""
    parts = []
    for action in state.action_history:
        if action.action_type == ActionType.INTERNAL_NOTE and action.message:
            parts.append(action.message)
    return " ".join(parts)


def _triage_action(state: EpisodeState) -> Tuple[bool, bool]:
    """Returns (priority_correct, category_correct)."""
    for action in state.action_history:
        if action.action_type == ActionType.TRIAGE:
            p_ok = str(action.priority).lower() == str(state.correct_priority).lower()
            c_ok = str(action.category).lower() == str(state.correct_category).lower()
            return p_ok, c_ok
    return False, False


def _escalated_to(state: EpisodeState, team: str) -> bool:
    for action in state.action_history:
        if action.action_type == ActionType.ESCALATE:
            if action.escalation_team and team.lower() in str(action.escalation_team).lower():
                return True
    return False


def _message_quality_score(text: str, min_words: int = 20, ideal_words: int = 80) -> float:
    """Simple quality proxy: penalize very short or very long messages."""
    words = len(text.split())
    if words < min_words:
        return max(0.0, words / min_words)
    if words > ideal_words * 3:
        return 0.6  # Too verbose penalty
    return 1.0


# ─────────────────────────────────────────────
# TASK 1 GRADER — Easy
# ─────────────────────────────────────────────

def grade_task_1(state: EpisodeState) -> Dict[str, Any]:
    """
    Grade: Billing duplicate charge inquiry.
    Max score: 1.0
    """
    rubric = {
        "correct_triage": 0.25,
        "customer_acknowledged": 0.20,
        "info_requested_or_escalated": 0.25,
        "resolution_quality": 0.30,
    }
    earned: Dict[str, float] = {}

    # 1. Correct triage (priority=medium, category=billing)
    p_ok, c_ok = _triage_action(state)
    triage_score = (0.5 if p_ok else 0.0) + (0.5 if c_ok else 0.0)
    earned["correct_triage"] = triage_score * rubric["correct_triage"]

    # 2. Customer acknowledged (empathy / apology for the issue)
    agent_text = _get_all_agent_messages(state)
    ack_keywords = ["apologize", "sorry", "understand", "inconvenience", "certainly", "happy to help", "look into"]
    acknowledged = _text_contains_any(agent_text, ack_keywords)
    earned["customer_acknowledged"] = rubric["customer_acknowledged"] * (1.0 if acknowledged else 0.0)

    # 3. Info requested or escalated to refunds team
    info_keywords = ["transaction", "statement", "bank", "date", "confirm", "provide", "screenshot"]
    info_requested = state.info_requested or _text_contains_any(agent_text, info_keywords)
    escalated_refunds = _escalated_to(state, "refunds")
    info_or_escalated = info_requested or escalated_refunds
    earned["info_requested_or_escalated"] = rubric["info_requested_or_escalated"] * (1.0 if info_or_escalated else 0.0)

    # 4. Resolution quality: resolved or escalated with relevant content
    res_keywords = ["refund", "duplicate", "charge", "credit", "process", "investigate"]
    has_resolution_content = _text_contains_any(agent_text, res_keywords)
    resolved_or_escalated = state.resolved or state.escalated
    quality = _message_quality_score(agent_text, min_words=15, ideal_words=60)
    res_score = 0.0
    if has_resolution_content and resolved_or_escalated:
        res_score = 1.0 * quality
    elif has_resolution_content:
        res_score = 0.6 * quality
    elif resolved_or_escalated:
        res_score = 0.3
    earned["resolution_quality"] = rubric["resolution_quality"] * res_score

    total = sum(earned.values())

    return {
        "score": round(min(total, 1.0), 4),
        "max_score": 1.0,
        "breakdown": earned,
        "rubric_weights": rubric,
        "diagnostics": {
            "triage_priority_correct": p_ok,
            "triage_category_correct": c_ok,
            "acknowledged_customer": acknowledged,
            "info_requested": info_requested,
            "escalated_to_refunds": escalated_refunds,
            "has_resolution_content": has_resolution_content,
            "message_quality": quality,
            "total_steps_used": state.step_number,
        },
    }


# ─────────────────────────────────────────────
# TASK 2 GRADER — Medium
# ─────────────────────────────────────────────

def grade_task_2(state: EpisodeState) -> Dict[str, Any]:
    """
    Grade: Technical crash — missing diagnostic info.
    Max score: 1.0
    """
    rubric = {
        "correct_triage": 0.20,
        "empathy_for_urgency": 0.15,
        "diagnostic_info_requested": 0.25,
        "workaround_offered": 0.15,
        "escalation_to_correct_team": 0.25,
    }
    earned: Dict[str, float] = {}
    agent_text = _get_all_agent_messages(state)

    # 1. Correct triage (priority=high, category=technical)
    p_ok, c_ok = _triage_action(state)
    triage_score = (0.5 if p_ok else 0.0) + (0.5 if c_ok else 0.0)
    earned["correct_triage"] = triage_score * rubric["correct_triage"]

    # 2. Empathy for urgency (presentation tomorrow)
    urgency_keywords = [
        "understand", "urgent", "presentation", "tomorrow", "important",
        "priority", "right away", "immediately", "asap"
    ]
    empathy_score = 1.0 if _text_contains_any(agent_text, urgency_keywords) else 0.0
    earned["empathy_for_urgency"] = rubric["empathy_for_urgency"] * empathy_score

    # 3. Diagnostic info requested — need OS, version, file info
    diag_keywords = ["operating system", "os", "windows", "mac", "version", "app version",
                     "file size", "file type", "steps to reproduce", "error message",
                     "screenshot", "device", "browser"]
    # Count distinct categories of info requested
    os_requested = _text_contains_any(agent_text, ["os", "operating system", "windows", "mac", "linux"])
    version_requested = _text_contains_any(agent_text, ["version", "app version", "build"])
    file_requested = _text_contains_any(agent_text, ["file", "size", "type", "format"])
    diag_score = (os_requested + version_requested + file_requested) / 3.0
    earned["diagnostic_info_requested"] = rubric["diagnostic_info_requested"] * diag_score

    # 4. Workaround offered
    workaround_keywords = [
        "workaround", "alternatively", "meanwhile", "try", "download manually",
        "export as", "google drive", "alternative format", "different browser",
        "different format", "word", "powerpoint", "png", "jpeg"
    ]
    workaround = _text_contains_any(agent_text, workaround_keywords)
    earned["workaround_offered"] = rubric["workaround_offered"] * (1.0 if workaround else 0.0)

    # 5. Escalation to senior tech
    escalated_senior = _escalated_to(state, "senior_tech")
    earned["escalation_to_correct_team"] = rubric["escalation_to_correct_team"] * (1.0 if escalated_senior else 0.0)

    total = sum(earned.values())

    return {
        "score": round(min(total, 1.0), 4),
        "max_score": 1.0,
        "breakdown": earned,
        "rubric_weights": rubric,
        "diagnostics": {
            "triage_priority_correct": p_ok,
            "triage_category_correct": c_ok,
            "empathy_shown": bool(empathy_score),
            "os_requested": os_requested,
            "version_requested": version_requested,
            "file_info_requested": file_requested,
            "workaround_offered": workaround,
            "escalated_to_senior_tech": escalated_senior,
            "total_steps_used": state.step_number,
        },
    }


# ─────────────────────────────────────────────
# TASK 3 GRADER — Hard
# ─────────────────────────────────────────────

def grade_task_3(state: EpisodeState) -> Dict[str, Any]:
    """
    Grade: Angry enterprise customer, policy edge case, SLA breach.
    Max score: 1.0
    """
    rubric = {
        "correct_triage_critical": 0.15,
        "acknowledgment_without_deflection": 0.20,
        "de_escalation_quality": 0.20,
        "escalation_to_management": 0.15,
        "concrete_commitment": 0.15,
        "internal_note_added": 0.10,
        "policy_override_recognized": 0.05,
    }
    earned: Dict[str, float] = {}
    agent_text = _get_all_agent_messages(state)
    internal_notes = _get_internal_notes(state)

    # 1. Correct triage (CRITICAL, REFUND)
    p_ok, c_ok = _triage_action(state)
    triage_score = (0.5 if p_ok else 0.0) + (0.5 if c_ok else 0.0)
    earned["correct_triage_critical"] = triage_score * rubric["correct_triage_critical"]

    # 2. Acknowledgment WITHOUT deflection
    # Must own the failure, not blame the customer or make excuses
    owns_failure = _text_contains_any(agent_text, [
        "sincerely apologize", "deeply sorry", "unacceptable", "we have failed",
        "take full responsibility", "should not have happened", "let you down",
        "completely understand", "rightfully frustrated", "this is on us"
    ])
    deflects = _text_contains_any(agent_text, [
        "unfortunately our policy", "per our terms", "as per policy",
        "I understand you're upset but", "I hear you however",
        "there's nothing I can do"
    ])
    ack_score = 0.0
    if owns_failure and not deflects:
        ack_score = 1.0
    elif owns_failure and deflects:
        ack_score = 0.4
    elif not owns_failure and not deflects:
        ack_score = 0.2
    earned["acknowledgment_without_deflection"] = rubric["acknowledgment_without_deflection"] * ack_score

    # 3. De-escalation quality
    # Empathy + validation + personal commitment
    de_esc_keywords = [
        "personally", "make this right", "top priority", "immediately",
        "understand", "frustration", "apologize", "you deserve", "urgent",
        "expedite", "direct contact", "dedicated"
    ]
    de_esc_count = sum(1 for kw in de_esc_keywords if kw in agent_text.lower())
    de_esc_score = min(de_esc_count / 5.0, 1.0)
    # Penalize if agent is defensive or cold
    cold_keywords = ["as I mentioned", "as stated", "actually", "technically", "however"]
    if _text_contains_any(agent_text, cold_keywords):
        de_esc_score *= 0.7
    earned["de_escalation_quality"] = rubric["de_escalation_quality"] * de_esc_score

    # 4. Escalation to management
    escalated_mgmt = _escalated_to(state, "management")
    earned["escalation_to_management"] = rubric["escalation_to_management"] * (1.0 if escalated_mgmt else 0.0)

    # 5. Concrete commitment (specific timeframe + callback)
    time_patterns = [
        r'\b\d+\s*(hour|hours|hr|hrs)\b',
        r'\btoday\b', r'\bwithin.*hour', r'\bby.*\d+(am|pm)',
        r'\bimmediately\b', r'\bright away\b'
    ]
    has_time_commitment = any(re.search(pat, agent_text.lower()) for pat in time_patterns)
    callback_promised = _text_contains_any(agent_text, [
        "callback", "call you", "reach out", "contact you", "follow up",
        "get back to you", "touch base"
    ])
    commitment_score = (0.5 if has_time_commitment else 0.0) + (0.5 if callback_promised else 0.0)
    earned["concrete_commitment"] = rubric["concrete_commitment"] * commitment_score

    # 6. Internal note added (must be substantive)
    has_internal_note = bool(state.action_history and any(
        a.action_type == ActionType.INTERNAL_NOTE for a in state.action_history
    ))
    note_substantive = len(internal_notes.split()) >= 20 if internal_notes else False
    note_score = 1.0 if (has_internal_note and note_substantive) else (0.4 if has_internal_note else 0.0)
    earned["internal_note_added"] = rubric["internal_note_added"] * note_score

    # 7. Policy override recognized
    # Agent should acknowledge that enterprise tier / SLA breach overrides standard refund limits
    override_keywords = [
        "enterprise", "sla", "policy exception", "override", "authorize",
        "special approval", "expedited", "waive", "bypass", "given the circumstances"
    ]
    override_internal = _text_contains_any(internal_notes, override_keywords)
    override_response = _text_contains_any(agent_text, override_keywords)
    override_score = 1.0 if (override_internal or override_response) else 0.0
    earned["policy_override_recognized"] = rubric["policy_override_recognized"] * override_score

    total = sum(earned.values())

    return {
        "score": round(min(total, 1.0), 4),
        "max_score": 1.0,
        "breakdown": earned,
        "rubric_weights": rubric,
        "diagnostics": {
            "triage_priority_correct": p_ok,
            "triage_category_correct": c_ok,
            "owns_failure": owns_failure,
            "deflects": deflects,
            "de_escalation_score": de_esc_score,
            "escalated_to_management": escalated_mgmt,
            "has_time_commitment": has_time_commitment,
            "callback_promised": callback_promised,
            "internal_note_added": has_internal_note,
            "note_substantive": note_substantive,
            "policy_override_recognized": bool(override_score),
            "total_steps_used": state.step_number,
        },
    }


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

GRADERS = {
    "task_1_billing_duplicate": grade_task_1,
    "task_2_technical_crash": grade_task_2,
    "task_3_enterprise_escalation": grade_task_3,
}


def grade(task_id: str, state: EpisodeState) -> Dict[str, Any]:
    """Run the appropriate grader for the given task."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task_id: {task_id}")
    return GRADERS[task_id](state)
