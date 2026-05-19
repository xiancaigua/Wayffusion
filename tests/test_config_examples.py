from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "configs" / "examples"


EXPECTED_FILES = {
    "env_template.yaml": ("num_agents", "reward_weights"),
    "policy_baseline_template.yaml": ("type", "policy_name"),
    "policy_bc_template.yaml": ("algorithm", "policy_class"),
    "policy_ppo_template.yaml": ("algorithm", "total_updates"),
    "policy_sac_template.yaml": ("algorithm", "total_steps"),
    "policy_td3_template.yaml": ("algorithm", "policy_noise"),
    "eval_single_task_template.yaml": ("mode", "tasks"),
    "eval_multitask_template.yaml": ("mode", "policies"),
    "eval_generalization_template.yaml": ("mode", "train_env_config"),
    "eval_learning_baselines_template.yaml": ("algorithms", "architectures"),
    "eval_scaling_fixed_n_template.yaml": ("protocol", "agent_counts"),
    "eval_scaling_variable_n_template.yaml": ("protocol", "train_sets"),
    "eval_ablation_observation_template.yaml": ("observation_modes", "output_path"),
    "eval_ablation_architecture_template.yaml": ("architectures", "output_path"),
    "eval_algorithm_comparison_template.yaml": ("algorithms", "output_path"),
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_all_example_files_exist():
    assert EXAMPLES_DIR.exists()
    actual = {path.name for path in EXAMPLES_DIR.glob("*.yaml")}
    assert actual == set(EXPECTED_FILES)


def test_example_files_parse_and_expose_required_keys():
    for filename, required_keys in EXPECTED_FILES.items():
        payload = load_yaml(EXAMPLES_DIR / filename)
        assert isinstance(payload, dict), filename
        for key in required_keys:
            assert key in payload, f"{filename} missing {key}"


def test_example_policy_algorithms_match_filename_category():
    assert load_yaml(EXAMPLES_DIR / "policy_bc_template.yaml")["algorithm"] == "bc"
    assert load_yaml(EXAMPLES_DIR / "policy_ppo_template.yaml")["algorithm"] == "ppo"
    assert load_yaml(EXAMPLES_DIR / "policy_sac_template.yaml")["algorithm"] == "sac"
    assert load_yaml(EXAMPLES_DIR / "policy_td3_template.yaml")["algorithm"] == "td3"
