# Undetermined v2 reward — backup before literal-relax / efficiency edit (2026-04-22)

This folder stores a **read-only snapshot** of the v2 undetermined reward logic and related CLI/floors **before** the following project changes:

- Post–type-2-success scaling of `(gx, gy)` shaping (`undetermined_v2_sl_post_success_literal_scale`)
- Post-success `S_L` dense/delta scaling (`undetermined_v2_sl_post_success_sl_shaping_scale`)
- Optional success bonus only on threshold crossing + sustain (`success_only_on_crossing`, `success_sustain_frac`)
- Pre-success step / travel penalties
- `apply_undetermined_v2_reward_floors` extensions (`type2_formation_efficiency`, `pattern_first` literal defaults)
- Extra `reward_terms` / CSV keys and `train.py` log fields

## Files

| File | Purpose |
|------|---------|
| `reward_calculator_legacy_methods.py` | Valid Python snapshot: legacy methods live on `_BackupRewardCalculatorV2Legacy_20260422_DO_NOT_IMPORT` (copy those two methods onto `RewardCalculator` in `reward_calculator.py`). |
| `config_cli_and_floors_legacy.md` | What to remove from `config/config.py` / `train.py` / `runner/shared/env_runner.py` to match the old surface area. |

## Restore (manual)

1. Open `MAPPO/envs/utils/reward_calculator.py`, replace the two methods with the bodies from `reward_calculator_legacy_methods.py` (keep them as **methods** on `RewardCalculator`, same indentation as surrounding methods).
2. Revert `config/config.py`, `train.py`, and `env_runner.py` per `config_cli_and_floors_legacy.md`.
3. Prefer `git checkout -- <file>` on those paths if your repo still has the pre-edit commit.

Do **not** add this directory to `PYTHONPATH`; these files are documentation / restore aids only.
