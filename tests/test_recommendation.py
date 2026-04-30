"""Tests for the material recommendation engine (inverse SE solver)."""
import pytest
from src.physics.recommendation import (
    MaterialConstraints,
    MaterialRecommendation,
    recommend_materials,
    pareto_filter,
)


class TestRecommendMaterials:
    """Tests for the recommend_materials function."""

    def test_returns_recommendations(self):
        """Should return between 1 and n_results recommendations."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=40,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
            n_results=5,
        )
        assert len(recs) > 0 and len(recs) <= 5

    def test_recommendations_meet_target(self):
        """Each recommendation should meet the target SE or be marked as not meeting it."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=40,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
        )
        for r in recs:
            assert r.achieved_se_db >= 40 or r.meets_target is False

    def test_copper_always_recommended(self):
        """Copper (excellent conductor) should always appear in recommendations."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=50,
                frequency_hz=1e9,
                max_thickness_m=2e-3,
            ),
            n_results=10,
        )
        names = [r.material_name for r in recs]
        has_copper = any("Cu" in n or "Copper" in n or "copper" in n for n in names)
        assert has_copper

    def test_weight_constraint_filters(self):
        """Density constraint should filter heavy materials from meeting-target results."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=30,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
                max_density_kg_m3=4000,
            ),
        )
        for r in recs:
            if r.meets_target:
                assert r.density_kg_m3 <= 4000

    def test_all_fields_populated(self):
        """All recommendation fields should be populated with reasonable values."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=30,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
            n_results=3,
        )
        for r in recs:
            assert r.material_name != ""
            assert r.achieved_se_db > 0
            assert r.optimal_thickness_m > 0
            assert r.density_kg_m3 > 0
            assert r.conductivity_s_m > 0
            assert r.skin_depth_m > 0
            assert r.explanation != ""

    def test_sorted_by_target_first(self):
        """Materials meeting the target should come before those that don't."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=30,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
            n_results=10,
        )
        # Find the transition point: once meets_target is False, all subsequent should be False
        found_non_meeting = False
        for r in recs:
            if not r.meets_target:
                found_non_meeting = True
            if found_non_meeting:
                assert r.meets_target is False

    def test_n_results_limits_output(self):
        """Output length should respect the n_results parameter."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=20,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
            n_results=3,
        )
        assert len(recs) <= 3

    def test_impossible_target_returns_best_effort(self):
        """Very high target with thin max and low density should still return best-effort results."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=500,
                frequency_hz=1e9,
                max_thickness_m=0.1e-3,
                # Restrict to lightweight materials only -- excludes high-perm alloys
                max_density_kg_m3=1000,
            ),
            n_results=5,
        )
        # Should return results even if none meet the extreme target
        assert len(recs) > 0
        # With such an extreme target and weight limit, none should meet it
        for r in recs:
            assert r.meets_target is False

    def test_se_margin_correct(self):
        """SE margin should equal achieved_se minus target_se."""
        target = 40
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=target,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
            n_results=5,
        )
        for r in recs:
            assert abs(r.se_margin_db - (r.achieved_se_db - target)) < 0.01


class TestParetoFilter:
    """Tests for the pareto_filter function."""

    def test_filters_dominated_solutions(self):
        """A solution dominated on all three objectives should be removed."""
        recs = [
            MaterialRecommendation(
                material_name="A",
                achieved_se_db=60,
                optimal_thickness_m=1e-3,
                density_kg_m3=3000,
                se_margin_db=20,
                meets_target=True,
            ),
            MaterialRecommendation(
                material_name="B",
                achieved_se_db=50,
                optimal_thickness_m=2e-3,
                density_kg_m3=8000,
                se_margin_db=10,
                meets_target=True,
            ),
        ]
        filtered = pareto_filter(recs)
        names = [r.material_name for r in filtered]
        assert "A" in names
        # B is dominated by A on all three: worse margin, thicker, denser
        assert "B" not in names

    def test_keeps_non_dominated(self):
        """Non-dominated solutions should all be kept."""
        recs = [
            MaterialRecommendation(
                material_name="A",
                achieved_se_db=60,
                optimal_thickness_m=2e-3,
                density_kg_m3=8000,
                se_margin_db=20,
                meets_target=True,
            ),
            MaterialRecommendation(
                material_name="B",
                achieved_se_db=50,
                optimal_thickness_m=1e-3,
                density_kg_m3=3000,
                se_margin_db=10,
                meets_target=True,
            ),
        ]
        filtered = pareto_filter(recs)
        names = [r.material_name for r in filtered]
        # Neither dominates the other: A has better SE margin, B is thinner and lighter
        assert "A" in names
        assert "B" in names

    def test_empty_input(self):
        """Empty list should return empty list."""
        assert pareto_filter([]) == []

    def test_single_item(self):
        """Single item should always be returned."""
        recs = [
            MaterialRecommendation(
                material_name="A",
                achieved_se_db=60,
                optimal_thickness_m=1e-3,
                density_kg_m3=3000,
                se_margin_db=20,
                meets_target=True,
            ),
        ]
        filtered = pareto_filter(recs)
        assert len(filtered) == 1
        assert filtered[0].material_name == "A"
