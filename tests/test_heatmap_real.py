import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestHeatmapRealPhysics:
    """Verify heatmap uses real EMICalculator, not mock math."""

    def test_copper_heatmap_returns_grid(self):
        resp = client.post("/api/v1/heatmap/generate", json={
            "composition": {"Cu": 100.0},
            "freq_start_mhz": 100,
            "freq_end_mhz": 1000,
            "thickness_start_mm": 0.1,
            "thickness_end_mm": 2.0,
            "num_points": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["frequencies_mhz"]) == 5
        assert len(data["thicknesses_mm"]) == 5
        assert len(data["se_matrix_db"]) == 5
        assert len(data["se_matrix_db"][0]) == 5

    def test_copper_se_values_are_physically_correct(self):
        """Copper 1mm at 1 GHz: skin depth ~2um, t/delta ~500, SE >> 100 dB."""
        resp = client.post("/api/v1/heatmap/generate", json={
            "composition": {"Cu": 100.0},
            "freq_start_mhz": 1000,
            "freq_end_mhz": 1000,
            "thickness_start_mm": 1.0,
            "thickness_end_mm": 1.0,
            "num_points": 1,
        })
        data = resp.json()
        se = data["se_matrix_db"][0][0]
        assert se > 100, f"Copper 1mm at 1 GHz should be >100 dB, got {se}"

    def test_thicker_shield_gives_higher_se(self):
        """SE must increase monotonically with thickness (no sin oscillations)."""
        resp = client.post("/api/v1/heatmap/generate", json={
            "composition": {"Cu": 100.0},
            "freq_start_mhz": 500,
            "freq_end_mhz": 500,
            "thickness_start_mm": 0.1,
            "thickness_end_mm": 2.0,
            "num_points": 10,
        })
        data = resp.json()
        # se_matrix_db[i][j] = SE at thickness i, frequency j
        # Extract SE at fixed frequency (col 0) across increasing thicknesses
        se_col = [row[0] for row in data["se_matrix_db"]]
        for i in range(1, len(se_col)):
            assert se_col[i] >= se_col[i - 1], (
                f"SE must increase with thickness: {se_col}"
            )
