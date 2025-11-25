"""Tests for hypersteer/data/utils.py"""

import pytest

from hypersteer.data.utils import get_intervention_locations, parse_positions


class TestParsePositions:
    """Tests for parse_positions function."""

    def test_parse_first_and_last(self):
        """Test parsing 'f2+l3' returns (2, 3)."""
        first_n, last_n = parse_positions("f2+l3")
        assert first_n == 2
        assert last_n == 3

    def test_parse_first_only(self):
        """Test parsing 'f5' returns (5, 0)."""
        first_n, last_n = parse_positions("f5")
        assert first_n == 5
        assert last_n == 0

    def test_parse_last_only(self):
        """Test parsing 'l10' returns (0, 10)."""
        first_n, last_n = parse_positions("l10")
        assert first_n == 0
        assert last_n == 10

    def test_parse_empty_positions(self):
        """Test parsing string without f or l returns (0, 0)."""
        first_n, last_n = parse_positions("other")
        assert first_n == 0
        assert last_n == 0


class TestGetInterventionLocations:
    """Tests for get_intervention_locations function."""

    def test_shared_weights_basic(self):
        """Test intervention locations with share_weights=True."""
        locations = get_intervention_locations(
            last_position=10,
            first_n=2,
            last_n=2,
            num_interventions=4,
            share_weights=True,
        )
        # With share_weights=True, all interventions have same locations
        assert len(locations) == 4
        assert all(loc == locations[0] for loc in locations)
        # Should include first 2 and last 2 positions
        assert 0 in locations[0]
        assert 1 in locations[0]
        assert 8 in locations[0]
        assert 9 in locations[0]

    def test_separate_left_right(self):
        """Test intervention locations with share_weights=False."""
        locations = get_intervention_locations(
            last_position=10,
            first_n=2,
            last_n=2,
            num_interventions=4,
            share_weights=False,
        )
        # First half should have left positions, second half right positions
        assert len(locations) == 4
        left_locs = locations[: len(locations) // 2]
        right_locs = locations[len(locations) // 2 :]

        # Left should start from 0
        assert 0 in left_locs[0]
        # Right should end at last_position - 1
        assert 9 in right_locs[0]

    def test_padding_first_mode(self):
        """Test padding with pad_mode='first' uses -1."""
        locations = get_intervention_locations(
            last_position=4,  # Small position to trigger padding
            first_n=5,
            last_n=0,
            num_interventions=2,
            share_weights=True,
            pad_mode="first",
        )
        # Should have -1 padding values
        assert -1 in locations[0]

    def test_padding_last_mode(self):
        """Test padding with pad_mode='last' uses last_position."""
        locations = get_intervention_locations(
            last_position=4,
            first_n=5,
            last_n=0,
            num_interventions=2,
            share_weights=True,
            pad_mode="last",
        )
        # Should have last_position (4) as padding
        assert 4 in locations[0]

    def test_with_positions_string(self):
        """Test using positions string parameter."""
        locations = get_intervention_locations(
            last_position=10,
            positions="f3+l2",
            num_interventions=2,
            share_weights=True,
        )
        assert len(locations) == 2
        # Should include first 3 and last 2 positions
        expected_positions = {0, 1, 2, 8, 9}
        actual_positions = set(locations[0])
        assert expected_positions.issubset(actual_positions)
