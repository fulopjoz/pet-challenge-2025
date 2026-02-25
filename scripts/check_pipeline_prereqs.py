#!/usr/bin/env python3
"""
Preflight checker for PET Challenge 2025 pipeline prerequisites.

Reports:
  1) Missing Python dependencies used by pipeline scripts
  2) Missing pipeline artifacts by stage

Exit codes:
  0 = ready
  1 = missing dependencies
  2 = missing artifacts
  3 = missing dependencies and artifacts
"""

import os
import sys
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def check_dependencies():
    required_modules = [
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "esm",
        "transformers",
        "propka",
        "pKAI",
        "sklearn",
        "xgboost",
    ]
    missing = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return required_modules, missing


def _existing(paths):
    return [p for p in paths if os.path.exists(os.path.join(PROJECT_ROOT, p))]


def check_artifacts():
    stage_requirements = {
        "inputs": {
            "all_of": [
                "data/petase_challenge_data/pet-2025-wildtype-cds.csv",
                "data/petase_challenge_data/predictive-pet-zero-shot-test-2025.csv",
            ]
        },
        "scores": {
            "any_of": [
                "results/esm2_scores.csv",
                "results/esmc_scores.csv",
            ]
        },
        "features": {
            "all_of": [
                "results/mutation_features.csv",
            ]
        },
        "submission": {
            "any_of": [
                "results/submission_zero_shot_v2.csv",
                "results/submission_zero_shot_v4.csv",
                "results/submission_zero_shot_v5.csv",
                "results/submission_zero_shot_v6.csv",
            ]
        },
        "validation": {
            "all_of": [
                "results/esm2_scores.csv",
                "results/submission_zero_shot_v2.csv",
            ]
        },
    }

    stage_report = {}
    missing_any_stage = False
    for stage, req in stage_requirements.items():
        all_of = req.get("all_of", [])
        any_of = req.get("any_of", [])
        missing_all_of = [p for p in all_of if not os.path.exists(os.path.join(PROJECT_ROOT, p))]
        existing_any_of = _existing(any_of) if any_of else []
        any_of_ok = (len(any_of) == 0) or (len(existing_any_of) > 0)
        stage_ok = (len(missing_all_of) == 0) and any_of_ok
        if not stage_ok:
            missing_any_stage = True
        stage_report[stage] = {
            "ok": stage_ok,
            "missing_all_of": missing_all_of,
            "any_of_candidates": any_of,
            "existing_any_of": existing_any_of,
        }
    return stage_report, missing_any_stage


def main():
    print("=" * 68)
    print("PET PIPELINE PRE-FLIGHT CHECK")
    print("=" * 68)

    deps_all, deps_missing = check_dependencies()
    stage_report, artifacts_missing = check_artifacts()

    print("\n[Dependencies]")
    print("Required modules: %s" % ", ".join(deps_all))
    if deps_missing:
        print("MISSING: %s" % ", ".join(deps_missing))
    else:
        print("OK: all required modules found")

    print("\n[Artifacts by Stage]")
    for stage, info in stage_report.items():
        status = "OK" if info["ok"] else "MISSING"
        print("- %s: %s" % (stage, status))
        if info["missing_all_of"]:
            print("    missing required files: %s" % info["missing_all_of"])
        if info["any_of_candidates"]:
            if info["existing_any_of"]:
                print("    found optional-set files: %s" % info["existing_any_of"])
            else:
                print("    need at least one of: %s" % info["any_of_candidates"])

    deps_flag = len(deps_missing) > 0
    if deps_flag and artifacts_missing:
        exit_code = 3
    elif deps_flag:
        exit_code = 1
    elif artifacts_missing:
        exit_code = 2
    else:
        exit_code = 0

    print("\nExit code: %d" % exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
