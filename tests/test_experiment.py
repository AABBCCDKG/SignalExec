import numpy as np
import pytest

from hail_ppo_reproduction import ExecutionConfig
from hail_ppo_reproduction.experiment import (
    ExperimentConfig, create_env, evaluate_policy,
)


def config(**kwargs):
    return ExperimentConfig(timesteps=256, eval_episodes=5, eval_batch_size=2,
                            steps=4, **kwargs)


def test_twap_and_immediate_complete_the_same_target():
    cfg = config()
    execution = ExecutionConfig()
    for name in ("twap", "immediate"):
        results = evaluate_policy(name, None, None, cfg, execution, 0.4)
        assert len(results) == 5
        assert [r["eval_seed"] for r in results] == list(range(cfg.eval_seed_start, cfg.eval_seed_start + 5))
        assert all(r["filled"] for r in results)


def test_forced_deadline_fill_has_an_economic_cost():
    cfg = config()
    execution = ExecutionConfig()
    natural = evaluate_policy("wait", None, None, cfg, execution, 0.4)
    forced = evaluate_policy("wait", None, None, cfg, execution, 0.4, True)
    assert all(r["fill_fraction"] == 0 and r["pnl"] == 0 for r in natural)
    assert all(r["fill_fraction"] == 1 and r["cost_bps"] > 0 for r in forced)


def test_evaluation_repeats_exactly_with_matched_day_seeds():
    args = ("hail_baseline", None, None, config(), ExecutionConfig(), 0.4)
    assert evaluate_policy(*args) == evaluate_policy(*args)


def test_recurrent_history_is_carried_then_reset_between_batches():
    class HistoryProbe:
        def __init__(self):
            self.calls = 0

        def predict(self, observations, state, episode_start, deterministic):
            step = self.calls % 4
            assert observations.ndim == 2
            assert np.all(episode_start == (step == 0))
            assert state == (None if step == 0 else step)
            assert deterministic
            self.calls += 1
            return np.zeros((len(observations), 1), dtype=np.float32), step + 1

    probe = HistoryProbe()
    evaluate_policy("memory_ppo", probe, 0, config(), ExecutionConfig(), 0.4)
    assert probe.calls == 12  # three batches, including the last single episode


def test_budget_rejects_unequal_ppo_rounding():
    with pytest.raises(ValueError, match="multiple of 256"):
        ExperimentConfig(timesteps=1000)


@pytest.mark.parametrize("algorithm", ["sac", "ppo", "memory_ppo"])
def test_model_save_reload_preserves_evaluation(tmp_path, algorithm):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from hail_ppo_reproduction import SyntheticMarketConfig
    from hail_ppo_reproduction.experiment import build_model
    torch.set_num_threads(1)
    cfg = config()
    execution = ExecutionConfig()
    env = create_env(SyntheticMarketConfig(steps=4), execution)
    model = build_model(algorithm, env, 0, 256)
    model.learn(total_timesteps=256)
    before = evaluate_policy(algorithm, model, 0, cfg, execution, 0.4)
    model.save(tmp_path / "policy.zip")
    loaded = type(model).load(tmp_path / "policy.zip", device="cpu")
    after = evaluate_policy(algorithm, loaded, 0, cfg, execution, 0.4)
    assert before == after
    env.close()
