"""Tests for the aggregate_metrics module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from aggregate_metrics import aggregate


class TestAggregate:
    """Tests for the aggregate function."""

    def _write_summary(self, directory, encoder, model, n_reps=3):
        """Write a fake summary CSV matching the real output format."""
        rows = []
        for rep in range(n_reps):
            rows.append({
                "encoder": encoder,
                "model": model,
                "spatial_effect": "SVC_X1_Raw",
                "repetition": rep,
                "pearson_r": 0.9 + np.random.RandomState(rep).rand() * 0.1,
                "ols_slope": 0.8 + np.random.RandomState(rep).rand() * 0.2,
                "rmse": 0.1 + np.random.RandomState(rep).rand() * 0.05,
            })
        df = pd.DataFrame(rows)
        path = os.path.join(directory, f"{encoder}_{model}_summary.csv")
        df.to_csv(path, index=False)

    def test_produces_output_files(self, tmp_path):
        self._write_summary(str(tmp_path), "NeRF", "MLP")
        self._write_summary(str(tmp_path), "rff", "MLP")
        aggregate(str(tmp_path))

        assert (tmp_path / "all_encoders_all_repetitions.csv").exists()
        assert (tmp_path / "all_encoders_summary_stats.csv").exists()

    def test_combined_row_count(self, tmp_path):
        self._write_summary(str(tmp_path), "NeRF", "MLP", n_reps=5)
        self._write_summary(str(tmp_path), "rff", "MLP", n_reps=5)
        aggregate(str(tmp_path))

        combined = pd.read_csv(tmp_path / "all_encoders_all_repetitions.csv")
        assert len(combined) == 10  # 5 reps x 2 encoders

    def test_summary_has_mean_std(self, tmp_path):
        self._write_summary(str(tmp_path), "NeRF", "XGBoost", n_reps=3)
        aggregate(str(tmp_path))

        stats = pd.read_csv(tmp_path / "all_encoders_summary_stats.csv")
        col_names = list(stats.columns)
        assert any("mean" in c for c in col_names)
        assert any("std" in c for c in col_names)

    def test_empty_directory(self, tmp_path, capsys):
        aggregate(str(tmp_path))
        captured = capsys.readouterr()
        assert "No summary files found" in captured.out
