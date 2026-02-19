"""Tests for core/softmin.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from softmatch_db.core.softmin import softmin, softmin_vec


class TestSoftmin:
    """Unit tests for the softmin function."""

    def test_identity(self) -> None:
        """softmin(1, 1) should equal 1."""
        assert math.isclose(softmin(1.0, 1.0), 1.0, rel_tol=1e-6)

    def test_symmetry(self) -> None:
        """softmin(a, b) == softmin(b, a)."""
        a, b = 0.7, 0.3
        assert math.isclose(softmin(a, b), softmin(b, a), rel_tol=1e-9)

    def test_upper_bound(self) -> None:
        """softmin(a, b) <= min(a, b)."""
        pairs = [(0.5, 0.8), (0.3, 0.3), (0.9, 0.1), (0.6, 0.6)]
        for a, b in pairs:
            assert softmin(a, b) <= min(a, b) + 1e-7

    def test_monotonicity(self) -> None:
        """Increasing either argument should not decrease the result."""
        base = softmin(0.5, 0.5)
        assert softmin(0.6, 0.5) >= base - 1e-7
        assert softmin(0.5, 0.6) >= base - 1e-7

    def test_low_values(self) -> None:
        """Both arguments near zero → result near zero."""
        result = softmin(0.01, 0.01)
        assert result < 0.05

    def test_high_values(self) -> None:
        """Both arguments near one → result near one."""
        result = softmin(0.99, 0.99)
        assert result > 0.95


class TestSoftminVec:
    """Unit tests for the vectorised softmin."""

    def test_matches_scalar(self) -> None:
        """Vector version should match scalar for each element."""
        a = 0.7
        b_arr = np.array([0.3, 0.5, 0.8, 1.0], dtype=np.float32)
        vec_result = softmin_vec(a, b_arr)
        for i, b in enumerate(b_arr):
            expected = softmin(a, float(b))
            assert math.isclose(float(vec_result[i]), expected, rel_tol=1e-5)

    def test_output_dtype(self) -> None:
        """Output should be float32."""
        result = softmin_vec(0.5, np.array([0.5], dtype=np.float32))
        assert result.dtype == np.float32
