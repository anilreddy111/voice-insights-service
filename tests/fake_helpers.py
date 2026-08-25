"""Aggregation math shared by FakeEngine (keeps prod/test behavior aligned)."""

import numpy as np


def aggregate(preds, f0_hz=None):
    if not preds:
        return {
            "gender_pred": "unknown", "gender_conf": 0.0,
            "bracket_pred": "unknown", "bracket_conf": 0.0,
            "age_median": None, "age_sigma": None,
            "n_windows": 0, "child_dominant": False, "reasons": [],
        }
    ages = np.array([p.age_years for p in preds])
    pf = float(np.mean([p.p_female for p in preds]))
    pm = float(np.mean([p.p_male for p in preds]))
    pc = float(np.mean([p.p_child for p in preds]))
    gender_pred = "female" if pf >= pm else "male"
    total = max(pf + pm + 0.3 * pc, 1e-9)
    conf = round(min(0.99, max(pf, pm) / total), 3)
    median_age = float(np.median(ages))
    bracket = (
        "18-30" if median_age < 30.5
        else "31-45" if median_age < 45.5
        else "46-60" if median_age < 60.5
        else "60+"
    )
    return {
        "gender_pred": gender_pred,
        "gender_conf": conf,
        "bracket_pred": bracket,
        "bracket_conf": 0.8,
        "age_median": round(median_age, 1),
        "age_sigma": 4.0,
        "n_windows": len(preds),
        "child_dominant": pc >= 0.55 and pc >= max(pf, pm),
        "reasons": [],
    }
