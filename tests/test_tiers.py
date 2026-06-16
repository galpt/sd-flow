"""Tests for tier definitions and sigma range segmentation."""

import pytest
from sd_flow.tiers import Tier, TIER_SIGMA_FRACTIONS, segment_sigma_range


class TestTierEnum:
    """Tier enum definition."""

    def test_has_four_members(self):
        members = list(Tier)
        assert len(members) == 4

    def test_members_are_correct(self):
        assert Tier.PRIORITY.value == 0
        assert Tier.NORMAL.value == 1
        assert Tier.LOW.value == 2
        assert Tier.DEFICIT.value == 3

    def test_enum_names(self):
        names = {m.name for m in Tier}
        assert names == {'PRIORITY', 'NORMAL', 'LOW', 'DEFICIT'}

    def test_ordering_by_value(self):
        values = [m.value for m in Tier]
        assert values == [0, 1, 2, 3]


class TestTierSigmaFractions:
    """TIER_SIGMA_FRACTIONS dictionary."""

    def test_covers_all_tiers(self):
        assert set(TIER_SIGMA_FRACTIONS.keys()) == {t for t in Tier}

    def test_priority_range(self):
        assert TIER_SIGMA_FRACTIONS[Tier.PRIORITY] == (0.75, 1.0)

    def test_normal_range(self):
        assert TIER_SIGMA_FRACTIONS[Tier.NORMAL] == (0.50, 0.75)

    def test_low_range(self):
        assert TIER_SIGMA_FRACTIONS[Tier.LOW] == (0.25, 0.50)

    def test_deficit_range(self):
        assert TIER_SIGMA_FRACTIONS[Tier.DEFICIT] == (0.0, 0.25)

    def test_fractions_cover_full_interval(self):
        lo_values = [TIER_SIGMA_FRACTIONS[t][0] for t in Tier]
        hi_values = [TIER_SIGMA_FRACTIONS[t][1] for t in Tier]
        assert min(lo_values) == 0.0
        assert max(hi_values) == 1.0


class TestSegmentSigmaRange:
    """segment_sigma_range() function."""

    def test_returns_four_segments(self):
        segments = segment_sigma_range(0.002, 80.0)
        assert len(segments) == 4

    def test_all_segments_are_tiers(self):
        segments = segment_sigma_range(0.002, 80.0)
        for tier, _, _ in segments:
            assert isinstance(tier, Tier)

    def test_segments_descending_order(self):
        """Segments must be in descending sigma order: PRIORITY first, DEFICIT last."""
        segments = segment_sigma_range(0.002, 80.0)
        tiers_in_order = [s[0] for s in segments]
        assert tiers_in_order == [Tier.PRIORITY, Tier.NORMAL, Tier.LOW, Tier.DEFICIT]

    def test_covers_full_range_no_gaps(self):
        """Each segment's hi equals the next segment's lo (or sigma_min)."""
        sigma_min, sigma_max = 0.002, 80.0
        segments = segment_sigma_range(sigma_min, sigma_max)
        # First segment hi == sigma_max
        assert segments[0][2] == sigma_max
        # Each segment's lo <= its hi (valid interval)
        for tier, lo, hi in segments:
            assert lo <= hi, f"{tier}: lo={lo} > hi={hi}"
        # Adjacent segments: segment[i].lo should equal segment[i+1].hi
        for i in range(len(segments) - 1):
            lo_current = segments[i][1]
            hi_next = segments[i + 1][2]
            assert lo_current == pytest.approx(hi_next), (
                f"Gap between {segments[i][0]} and {segments[i+1][0]}: "
                f"{segments[i][0]}.lo={lo_current} != {segments[i+1][0]}.hi={hi_next}"
            )
        # Last segment's lo == sigma_min
        assert segments[-1][1] == sigma_min

    def test_sigma_boundaries_default(self):
        sigma_min, sigma_max = 0.002, 80.0
        segments = segment_sigma_range(sigma_min, sigma_max)
        # PRIORITY: hi = 1.0 * 80 = 80, lo = max(0.75*80, 0.002) = 60
        assert segments[0] == (Tier.PRIORITY, 60.0, 80.0)
        # NORMAL: hi = 0.75*80 = 60, lo = max(0.5*80, 0.002) = 40
        assert segments[1] == (Tier.NORMAL, 40.0, 60.0)
        # LOW: hi = 0.5*80 = 40, lo = max(0.25*80, 0.002) = 20
        assert segments[2] == (Tier.LOW, 20.0, 40.0)
        # DEFICIT: hi = 0.25*80 = 20, lo = max(0.0*80, 0.002) = 0.002
        assert segments[3] == (Tier.DEFICIT, 0.002, 20.0)

    def test_different_sigma_max(self):
        sigma_min, sigma_max = 0.001, 10.0
        segments = segment_sigma_range(sigma_min, sigma_max)
        assert segments[0] == (Tier.PRIORITY, 7.5, 10.0)

    def test_different_sigma_min(self):
        sigma_min, sigma_max = 5.0, 80.0
        segments = segment_sigma_range(sigma_min, sigma_max)
        # PRIORITY: lo = max(0.75*80, 5.0) = max(60, 5) = 60
        assert segments[0] == (Tier.PRIORITY, 60.0, 80.0)
        # NORMAL: lo = max(0.5*80, 5.0) = max(40, 5) = 40
        assert segments[1] == (Tier.NORMAL, 40.0, 60.0)
        # LOW: lo = max(0.25*80, 5.0) = max(20, 5) = 20
        assert segments[2] == (Tier.LOW, 20.0, 40.0)
        # DEFICIT: lo = max(0.0*80, 5.0) = 5.0
        assert segments[3] == (Tier.DEFICIT, 5.0, 20.0)

    def test_sigma_min_larger_than_fraction_boundary(self):
        """When sigma_min exceeds a fraction boundary, sigma_lo is clamped to sigma_min."""
        sigma_min, sigma_max = 70.0, 80.0
        segments = segment_sigma_range(sigma_min, sigma_max)
        # PRIORITY: lo = max(0.75*80, 70) = 70, hi = 1.0*80 = 80
        assert segments[0][0] == Tier.PRIORITY
        assert segments[0][1] == 70.0
        assert segments[0][2] == 80.0
        # NORMAL: lo = max(0.5*80, 70) = 70, hi = 0.75*80 = 60
        assert segments[1][0] == Tier.NORMAL
        assert segments[1][1] == 70.0
        assert segments[1][2] == 60.0
        # LOW: lo = max(0.25*80, 70) = 70, hi = 0.5*80 = 40
        assert segments[2][0] == Tier.LOW
        assert segments[2][1] == 70.0
        assert segments[2][2] == 40.0
        # DEFICIT: lo = max(0.0*80, 70) = 70, hi = 0.25*80 = 20
        assert segments[3][0] == Tier.DEFICIT
        assert segments[3][1] == 70.0
        assert segments[3][2] == 20.0

    def test_sigma_min_zero(self):
        sigma_min, sigma_max = 0.0, 80.0
        segments = segment_sigma_range(sigma_min, sigma_max)
        assert segments[3] == (Tier.DEFICIT, 0.0, 20.0)

    def test_segments_have_correct_types(self):
        segments = segment_sigma_range(0.002, 80.0)
        for tier, lo, hi in segments:
            assert isinstance(tier, Tier)
            assert isinstance(lo, float)
            assert isinstance(hi, float)
