"""Tests for DispatchRotator."""

import pytest
from sd_flow.rotating_dispatch import DispatchRotator


class TestDispatchRotatorInit:
    """DispatchRotator construction and initial state."""

    def test_initial_phase_is_zero(self):
        rot = DispatchRotator()
        assert rot.phase == 0

    def test_initial_phase_property(self):
        rot = DispatchRotator()
        assert rot._phase == 0
        assert rot.phase == 0

    def test_n_tiers_default(self):
        rot = DispatchRotator()
        assert rot.n_tiers == 4


class TestDispatchRotatorCurrentOrder:
    """DispatchRotator.current_order()."""

    def test_phase_0_order(self):
        rot = DispatchRotator()
        order = rot.current_order()
        assert order == [0, 1, 2, 3]

    def test_all_phases_have_four_elements(self):
        for phase in range(4):
            rot = DispatchRotator()
            rot._phase = phase
            order = rot.current_order()
            assert len(order) == 4

    def test_each_phase_is_a_rotation(self):
        """Each phase is a rotation, so each tier appears exactly once."""
        for phase in range(4):
            rot = DispatchRotator()
            rot._phase = phase
            order = rot.current_order()
            assert sorted(order) == [0, 1, 2, 3]

    def test_phase_1_order(self):
        rot = DispatchRotator()
        rot._phase = 1
        assert rot.current_order() == [1, 2, 3, 0]

    def test_phase_2_order(self):
        rot = DispatchRotator()
        rot._phase = 2
        assert rot.current_order() == [2, 3, 0, 1]

    def test_phase_3_order(self):
        rot = DispatchRotator()
        rot._phase = 3
        assert rot.current_order() == [3, 0, 1, 2]

    def test_each_tier_starts_first_in_one_phase(self):
        """Each tier (0-3) appears first exactly once across the 4 phases."""
        first_positions = []
        for phase in range(4):
            rot = DispatchRotator()
            rot._phase = phase
            first_positions.append(rot.current_order()[0])
        assert sorted(first_positions) == [0, 1, 2, 3]


class TestDispatchRotatorAdvance:
    """DispatchRotator.advance()."""

    def test_advance_phase_0_to_1(self):
        rot = DispatchRotator()
        order = rot.advance()
        assert rot.phase == 1
        assert order == [1, 2, 3, 0]

    def test_advance_cycles_all_four_phases(self):
        rot = DispatchRotator()
        orders = []
        for _ in range(4):
            rot.advance()
            orders.append(rot.current_order())
        assert orders == [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ]

    def test_advance_returns_current_order(self):
        rot = DispatchRotator()
        result = rot.advance()
        assert result == rot.current_order()

    def test_after_4_advances_back_to_phase_0(self):
        rot = DispatchRotator()
        for _ in range(4):
            rot.advance()
        assert rot.phase == 0
        assert rot.current_order() == [0, 1, 2, 3]

    def test_advance_wraps_around_8_times(self):
        """After 8 advances, should be back to phase 0 (two full cycles)."""
        rot = DispatchRotator()
        for _ in range(8):
            rot.advance()
        assert rot.phase == 0

    def test_advance_does_not_mutate_phases(self):
        """PHASES class variable should remain unchanged after advance."""
        rot = DispatchRotator()
        original_phases = [list(p) for p in DispatchRotator.PHASES]
        for _ in range(10):
            rot.advance()
        assert DispatchRotator.PHASES == original_phases


class TestDispatchRotatorReset:
    """DispatchRotator.reset()."""

    def test_reset_returns_to_phase_0(self):
        rot = DispatchRotator()
        rot.advance()
        rot.advance()
        assert rot.phase == 2
        rot.reset()
        assert rot.phase == 0

    def test_reset_after_full_cycle(self):
        rot = DispatchRotator()
        for _ in range(4):
            rot.advance()
        assert rot.phase == 0
        rot.reset()
        assert rot.phase == 0

    def test_reset_order_is_phase_0(self):
        rot = DispatchRotator()
        rot.advance()
        rot.advance()
        rot.reset()
        assert rot.current_order() == [0, 1, 2, 3]


class TestDispatchRotatorCycle:
    """Full-cycle behavior of DispatchRotator."""

    def test_full_cycle_four_phases(self):
        """Each phase appears exactly once in a 4-advance cycle from phase 0."""
        rot = DispatchRotator()
        phases_seen = {rot.phase}
        for _ in range(4):
            rot.advance()
            phases_seen.add(rot.phase)
        assert phases_seen == {0, 1, 2, 3}

    def test_each_tier_first_exactly_once_per_cycle(self):
        """In a 4-phase cycle, each tier index should be first exactly once."""
        rot = DispatchRotator()
        first_tiers = [rot.current_order()[0]]  # phase 0
        for _ in range(4):
            rot.advance()
            first_tiers.append(rot.current_order()[0])
        # first_tiers = [0, 1, 2, 3, 0] — each tier first once plus wrap
        assert sorted(first_tiers[:-1]) == [0, 1, 2, 3]

    def test_consistent_through_multiple_cycles(self):
        """Multiple cycles should behave identically."""
        rot = DispatchRotator()
        # Two full cycles
        cycle1_orders = []
        for _ in range(4):
            cycle1_orders.append(rot.current_order())
            rot.advance()
        # Now at phase 0 again
        cycle2_orders = []
        for _ in range(4):
            cycle2_orders.append(rot.current_order())
            rot.advance()
        assert cycle1_orders == cycle2_orders
