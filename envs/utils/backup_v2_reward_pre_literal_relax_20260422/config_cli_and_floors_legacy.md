# Legacy surface area (remove to match pre-edit behavior)

## `config/config.py`

1. **Delete** the block of `parser.add_argument` entries from  
   `--undetermined_v2_type2_formation_efficiency` through  
   `--undetermined_v2_sl_pre_success_travel_penalty` (inclusive), i.e. the seven new arguments added after `--undetermined_v2_type2_pattern_first`.

2. In **`apply_undetermined_v2_reward_floors`**, **delete**:
   - The entire `if getattr(args, "undetermined_v2_type2_formation_efficiency", False):` block (preset defaults).
   - At the end of the `undetermined_v2_type2_pattern_first` branch, **delete** the nested block:
     ```python
     if not getattr(args, "undetermined_v2_type2_formation_efficiency", False):
         if float(getattr(args, "undetermined_v2_sl_post_success_literal_scale", 1.0)) > 0.999:
             args.undetermined_v2_sl_post_success_literal_scale = 0.28
         if float(getattr(args, "undetermined_v2_sl_post_success_sl_shaping_scale", 1.0)) > 0.999:
             args.undetermined_v2_sl_post_success_sl_shaping_scale = 0.42
     ```

## `train.py`

In the `print(...)` for undetermined v2 type-2 reward, **revert** to a single line ending with  
`type2_pattern_first=...` only (remove `type2_formation_efficiency`, `sl_post_lit`, `sl_post_shape`, `sl_succ_cross_only`, `sl_succ_sustain`, `sl_pre_step`, `sl_pre_trav`).

## `runner/shared/env_runner.py`

In **`REWARD_TERMS_CSV_COLUMNS`**, remove these four column names if present:

- `undetermined_v2_sl_type2_crossing`
- `undetermined_v2_sl_literal_relax_w`
- `undetermined_v2_sl_pre_success_step_raw`
- `undetermined_v2_sl_pre_success_travel_raw`
