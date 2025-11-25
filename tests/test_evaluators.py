"""Tests for hypersteer/evaluators/lm_judge.py"""

import pytest

from hypersteer.evaluators.lm_judge import LMJudgeEvaluator


@pytest.fixture
def evaluator():
    """Create a LMJudgeEvaluator instance for testing."""
    return LMJudgeEvaluator(model_name="test-model")


class TestGetRatingFromCompletion:
    """Tests for _get_rating_from_completion method."""

    def test_extracts_rating_basic(self, evaluator):
        """Test extracting rating from 'Rating: 1.5' format."""
        completion = "This is a good response. Rating: 1.5"
        rating = evaluator._get_rating_from_completion(completion)
        assert rating == 1.5

    def test_extracts_rating_with_brackets(self, evaluator):
        """Test extracting rating from 'Rating: [2.0]' format."""
        completion = "Analysis complete. Rating: [2.0]"
        rating = evaluator._get_rating_from_completion(completion)
        assert rating == 2.0

    def test_extracts_rating_with_asterisks(self, evaluator):
        """Test extracting rating from 'Rating: **1**' format."""
        completion = "Final verdict. Rating: **1**"
        rating = evaluator._get_rating_from_completion(completion)
        assert rating == 1.0

    def test_returns_default_when_no_rating(self, evaluator):
        """Test returns DEFAULT_RATING when no rating found."""
        completion = "This response has no rating marker."
        rating = evaluator._get_rating_from_completion(completion)
        assert rating == LMJudgeEvaluator.DEFAULT_RATING

    def test_handles_rating_with_period(self, evaluator):
        """Test extracting rating that ends with period."""
        completion = "Rating: 1.5."
        rating = evaluator._get_rating_from_completion(completion)
        assert rating == 1.5


class TestGetRatingsFromCompletions:
    """Tests for _get_ratings_from_completions method."""

    def test_extracts_multiple_ratings(self, evaluator):
        """Test extracting ratings from multiple completions."""
        completions = [
            "Good response. Rating: 1.5",
            "Great response. Rating: 2.0",
            "Average response. Rating: 1.0",
        ]
        ratings = evaluator._get_ratings_from_completions(completions)
        assert ratings == [1.5, 2.0, 1.0]

    def test_filters_out_of_range_values(self, evaluator):
        """Test ratings outside min/max are replaced with default."""
        completions = [
            "Rating: 5.0",  # Above max (2.0)
            "Rating: 1.5",  # Valid
            "Rating: -1.0",  # Below min (0.0)
        ]
        ratings = evaluator._get_ratings_from_completions(
            completions, min_rating=0.0, max_rating=2.0
        )
        assert ratings[0] == LMJudgeEvaluator.DEFAULT_RATING  # Out of range
        assert ratings[1] == 1.5  # Valid
        assert ratings[2] == LMJudgeEvaluator.DEFAULT_RATING  # Out of range

    def test_handles_parsing_errors(self, evaluator):
        """Test gracefully handles completions that can't be parsed."""
        completions = [
            "Rating: abc",  # Invalid - not a number
            "Rating: 1.5",  # Valid
        ]
        ratings = evaluator._get_ratings_from_completions(completions)
        assert ratings[0] == LMJudgeEvaluator.DEFAULT_RATING
        assert ratings[1] == 1.5


class TestLMJudgeEvaluatorInit:
    """Tests for LMJudgeEvaluator initialization."""

    def test_init_with_model_name(self):
        """Test evaluator initializes with model_name."""
        evaluator = LMJudgeEvaluator(model_name="test-model")
        assert evaluator.model_name == "test-model"

    def test_init_with_kwargs(self):
        """Test evaluator initializes with optional kwargs."""
        evaluator = LMJudgeEvaluator(
            model_name="test-model",
            concept_id=42,
        )
        assert evaluator.concept_id == 42

    def test_str_representation(self):
        """Test __str__ returns expected string."""
        evaluator = LMJudgeEvaluator(model_name="test-model")
        assert str(evaluator) == "LMJudgeEvaluator"
