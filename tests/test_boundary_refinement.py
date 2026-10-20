"""Exact refinement must replace complete match groups without reordering rows."""
import numpy as np
import pytest

from mlx_geospatial.join import _replace_refined_pairs


@pytest.mark.parametrize("seed", range(12))
def test_replace_refined_groups_matches_full_join_oracle(seed):
    rng = np.random.default_rng(seed)
    old = rng.random((80, 12)) < 0.3
    near = np.sort(rng.choice(80, size=25, replace=False))
    replacement = rng.random((len(near), 12)) < 0.25
    # Pin additions, deletions, and changes in overlap multiplicity.
    replacement[0] = False
    replacement[1] = True
    old[near[1]] = False
    expected = old.copy()
    expected[near] = replacement
    old_pair = np.nonzero(old)
    point, polygon = np.nonzero(replacement)
    permutation = rng.permutation(len(point))
    result = _replace_refined_pairs(old_pair, near, point[permutation], polygon[permutation])
    for actual, wanted in zip(result, np.nonzero(expected)):
        np.testing.assert_array_equal(actual, wanted)
        assert actual.dtype == np.int64


@pytest.mark.parametrize("old_hit,new_hit", [(False, False), (False, True), (True, False)])
def test_replace_refined_groups_empty_streams(old_hit, new_hit):
    old = np.full((4, 2), old_hit)
    replacement = np.full((4, 2), new_hit)
    result = _replace_refined_pairs(np.nonzero(old), np.arange(4), *np.nonzero(replacement))
    for actual, wanted in zip(result, np.nonzero(replacement)):
        np.testing.assert_array_equal(actual, wanted)


def test_confirmed_pairs_reuse_materialized_output():
    point = np.array([1, 1, 4, 5, 5, 8], dtype=np.int64)
    polygon = np.array([0, 2, 1, 0, 2, 1], dtype=np.int64)
    result = _replace_refined_pairs(
        (point, polygon), np.array([1, 5, 6]),
        np.array([1, 0, 1, 0]), np.array([2, 2, 0, 0]),
    )
    assert result[0] is point
    assert result[1] is polygon
