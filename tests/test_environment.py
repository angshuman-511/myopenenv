"""
Test suite for SupportTriage OpenEnv.
Tests: reset(), step(), state(), graders, reward shaping, action validation.

Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import SupportTriageEnv
from env.models import (
    Action, ActionType, Priority, Category, EscalationTeam,
    Observation, EpisodeState, Reward, StepResult
)
from graders.task_graders import grade
from tasks.task_definitions import list_tasks, TASKS


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def env():
    return SupportTriageEnv()


@pytest.fixture
def env_t1(env):
    env.reset("task_1_billing_duplicate")
    return env


@pytest.fixture
def env_t2(env):
    env.reset("task_2_technical_crash")
    return env


@pytest.fixture
def env_t3(env):
    env.reset("task_3_enterprise_escalation")
    return env


# ─────────────────────────────────────────────
# Test: Task Registry
# ─────────────────────────────────────────────

class TestTaskRegistry:
    def test_three_tasks_exist(self):
        tasks = list_tasks()
        assert len(tasks) == 3

    def test_difficulty_progression(self):
        difficulties = [t["difficulty"] for t in list_tasks()]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_invalid_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset("nonexistent_task")


# ─────────────────────────────────────────────
# Test: Reset
# ─────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("task_1_billing_duplicate")
        assert isinstance(obs, Observation)

    def test_reset_clears_state(self, env):
        env.reset("task_1_billing_duplicate")
        env.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        # Reset again
        obs = env.reset("task_1_billing_duplicate")
        assert obs.step_number == 0
        state = env.state()
        assert state.step_number == 0
        assert len(state.action_history) == 0
        assert state.cumulative_reward == 0.0

    def test_reset_observation_fields(self, env):
        obs = env.reset("task_2_technical_crash")
        assert obs.task_id == "task_2_technical_crash"
        assert obs.max_steps == 8
        assert obs.step_number == 0
        assert obs.ticket is not None
        assert obs.episode_id != ""
        assert len(obs.available_macros) > 0

    def test_reset_produces_clean_ticket(self, env):
        obs = env.reset("task_1_billing_duplicate")
        assert obs.ticket.priority is None
        assert obs.ticket.category is None


# ─────────────────────────────────────────────
# Test: Step
# ─────────────────────────────────────────────

class TestStep:
    def test_step_returns_step_result(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.TRIAGE,
            priority=Priority.MEDIUM,
            category=Category.BILLING
        ))
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, Reward)
        assert isinstance(result.done, bool)

    def test_step_increments_counter(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        assert env_t1.state().step_number == 1

    def test_triage_required_before_resolve(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.RESOLVE,
            resolution_summary="Resolved the billing issue."
        ))
        # Should get invalid action penalty
        assert result.reward.value < 0

    def test_resolve_ends_episode(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        env_t1.step(Action(action_type=ActionType.RESPOND, message="Thank you for reaching out. I will look into the duplicate charge."))
        result = env_t1.step(Action(
            action_type=ActionType.RESOLVE,
            resolution_summary="Investigated duplicate charge, escalating to refunds team for resolution."
        ))
        assert result.done

    def test_step_after_done_raises(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        env_t1.step(Action(action_type=ActionType.RESOLVE, resolution_summary="Done."))
        with pytest.raises(RuntimeError):
            env_t1.step(Action(action_type=ActionType.RESPOND, message="Another message"))

    def test_truncation_at_max_steps(self, env_t1):
        """Episode should truncate when max_steps reached without explicit resolve."""
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        result = None
        for _ in range(10):  # max_steps=6 for task 1
            if result and (result.done or result.truncated):
                break
            result = env_t1.step(Action(
                action_type=ActionType.RESPOND,
                message="Thank you for your patience, we are looking into this."
            ))
        assert result.done or result.truncated


# ─────────────────────────────────────────────
# Test: State
# ─────────────────────────────────────────────

class TestState:
    def test_state_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_returns_episode_state(self, env_t1):
        state = env_t1.state()
        assert isinstance(state, EpisodeState)

    def test_state_action_history_grows(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        assert len(env_t1.state().action_history) == 1

    def test_state_flags_update(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        assert env_t1.state().triage_done

        env_t1.step(Action(action_type=ActionType.RESPOND, message="I see you were charged twice. Let me look into this."))
        assert env_t1.state().responded


# ─────────────────────────────────────────────
# Test: Reward
# ─────────────────────────────────────────────

class TestReward:
    def test_correct_triage_positive_reward(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.TRIAGE,
            priority=Priority.MEDIUM,
            category=Category.BILLING
        ))
        assert result.reward.value > 0

    def test_wrong_triage_negative_reward(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.TRIAGE,
            priority=Priority.LOW,         # Wrong — should be MEDIUM
            category=Category.GENERAL      # Wrong — should be BILLING
        ))
        assert result.reward.value < 0

    def test_reward_in_valid_range(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING
        ))
        assert -1.0 <= result.reward.value <= 1.0

    def test_reward_has_components(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING
        ))
        assert isinstance(result.reward.components, dict)
        assert len(result.reward.components) > 0

    def test_invalid_action_penalty(self, env_t1):
        result = env_t1.step(Action(
            action_type=ActionType.TRIAGE,
            priority=None,    # Missing required field
            category=Category.BILLING
        ))
        assert result.reward.value < 0

    def test_repetition_penalty(self, env_t2):
        """Repeating same action type many times should incur penalty."""
        env_t2.step(Action(action_type=ActionType.TRIAGE, priority=Priority.HIGH, category=Category.TECHNICAL))
        rewards = []
        for _ in range(4):
            r = env_t2.step(Action(action_type=ActionType.RESPOND, message="We are looking into this issue right away."))
            rewards.append(r.reward.value)
        # Later rewards should be lower due to repetition penalty
        assert rewards[-1] < rewards[0]


# ─────────────────────────────────────────────
# Test: Graders
# ─────────────────────────────────────────────

class TestGraders:
    def _run_perfect_t1(self, env):
        """Run a near-perfect Task 1 episode."""
        env.reset("task_1_billing_duplicate")
        env.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        env.step(Action(action_type=ActionType.RESPOND,
            message="I sincerely apologize for the duplicate charge, Maria. I understand how frustrating this must be."))
        env.step(Action(action_type=ActionType.REQUEST_INFO,
            message="Could you please provide the date and transaction ID of the duplicate charge from your bank statement?"))
        env.step(Action(action_type=ActionType.ESCALATE,
            escalation_team=EscalationTeam.REFUNDS_TEAM,
            escalation_reason="Verified duplicate charge, needs refund processing"))
        env.step(Action(action_type=ActionType.RESOLVE,
            resolution_summary="Acknowledged duplicate charge, requested transaction confirmation, escalated to refunds team for processing within 3-5 business days."))

    def test_perfect_task1_high_score(self, env):
        self._run_perfect_t1(env)
        state = env.state()
        result = grade("task_1_billing_duplicate", state)
        assert result["score"] >= 0.70, f"Expected ≥0.70, got {result['score']}"

    def test_empty_task1_low_score(self, env):
        """Agent that only triages gets low score."""
        env.reset("task_1_billing_duplicate")
        env.step(Action(action_type=ActionType.TRIAGE, priority=Priority.LOW, category=Category.GENERAL))
        env.step(Action(action_type=ActionType.RESOLVE, resolution_summary="Done."))
        state = env.state()
        result = grade("task_1_billing_duplicate", state)
        assert result["score"] < 0.40

    def test_grader_score_in_range(self, env):
        env.reset("task_3_enterprise_escalation")
        env.step(Action(action_type=ActionType.TRIAGE, priority=Priority.CRITICAL, category=Category.REFUND))
        env.step(Action(action_type=ActionType.RESOLVE, resolution_summary="Resolved."))
        state = env.state()
        result = grade("task_3_enterprise_escalation", state)
        assert 0.0 <= result["score"] <= 1.0

    def test_grader_deterministic(self, env):
        """Same episode → same score."""
        env.reset("task_2_technical_crash")
        env.step(Action(action_type=ActionType.TRIAGE, priority=Priority.HIGH, category=Category.TECHNICAL))
        env.step(Action(action_type=ActionType.REQUEST_INFO,
            message="Could you please share your operating system, app version, and the file size you were trying to export?"))
        env.step(Action(action_type=ActionType.ESCALATE,
            escalation_team=EscalationTeam.SENIOR_TECH,
            escalation_reason="PDF export crash requiring technical investigation"))
        env.step(Action(action_type=ActionType.RESOLVE,
            resolution_summary="Gathered diagnostic info, escalated to senior tech, offered workaround."))
        state = env.state()
        result1 = grade("task_2_technical_crash", state)
        result2 = grade("task_2_technical_crash", state)
        assert result1["score"] == result2["score"]

    def test_grader_breakdown_keys_match_rubric(self, env):
        env.reset("task_1_billing_duplicate")
        env.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        env.step(Action(action_type=ActionType.RESOLVE, resolution_summary="Issue resolved after investigation."))
        state = env.state()
        result = grade("task_1_billing_duplicate", state)
        expected_keys = {"correct_triage", "customer_acknowledged", "info_requested_or_escalated", "resolution_quality"}
        assert set(result["breakdown"].keys()) == expected_keys


# ─────────────────────────────────────────────
# Test: Action Validation
# ─────────────────────────────────────────────

class TestActionValidation:
    def test_triage_missing_priority_penalized(self, env_t1):
        result = env_t1.step(Action(action_type=ActionType.TRIAGE, category=Category.BILLING))
        assert result.reward.value < 0

    def test_respond_with_empty_message_penalized(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        result = env_t1.step(Action(action_type=ActionType.RESPOND, message="Hi"))  # Too short
        assert result.reward.value == pytest.approx(-0.05, abs=0.01)

    def test_unknown_macro_penalized(self, env_t1):
        env_t1.step(Action(action_type=ActionType.TRIAGE, priority=Priority.MEDIUM, category=Category.BILLING))
        result = env_t1.step(Action(action_type=ActionType.APPLY_MACRO, macro_name="nonexistent_macro"))
        assert result.reward.value < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
