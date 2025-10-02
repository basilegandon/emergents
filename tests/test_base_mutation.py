"""
Comprehensive unit tests for emergents.mutations.base module.
Tests the base Mutation class and its abstract interface.
"""

import pytest  # pyright: ignore[reportUnusedImport]

from emergents.mutations.base import Mutation


class TestMutationBaseClass:
    """Test the base Mutation class."""

    def test_mutation_initialization(self):
        """Test Mutation initialization."""
        # Default initialization
        mutation = Mutation()
        assert mutation.rng_state is None

        # With RNG state
        mutation_with_state = Mutation(rng_state=12345)
        assert mutation_with_state.rng_state == 12345
