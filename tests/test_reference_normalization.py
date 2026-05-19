from __future__ import annotations

from envs.metrics import compute_reference_normalization


def test_reference_normalization_standard_ordering():
    result = compute_reference_normalization(return_value=5.0, random_return=0.0, heuristic_return=10.0)
    assert result["reference_order_flipped"] == 0.0
    assert result["reference_best_method"] == "heuristic"
    assert 0.0 <= result["normalized_score"] <= 1.0


def test_reference_normalization_flips_when_heuristic_is_worse_than_random():
    result = compute_reference_normalization(return_value=5.0, random_return=10.0, heuristic_return=0.0)
    assert result["reference_order_flipped"] == 1.0
    assert result["reference_best_method"] == "random"


def test_reference_normalization_marks_unstable_small_gap():
    result = compute_reference_normalization(return_value=1.0, random_return=1.0, heuristic_return=1.0000001)
    assert result["reference_unstable"] == 1.0


def test_reference_normalization_clips_stored_score_but_preserves_raw_value():
    result = compute_reference_normalization(return_value=100.0, random_return=0.0, heuristic_return=1.0)
    assert -5.0 <= result["normalized_score"] <= 5.0
    assert result["normalized_score_raw"] > 5.0
