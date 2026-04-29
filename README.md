Works based on *Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance*.

## Implemented

- Seperated Reward
- Seperated Dataset
- Data Generator

## TODO

- Faster Code

---

## Training

Run from repository root:

```bash
cd /path/to/MAPPO
python train.py [args...]
```

Outputs are written to:

- `results/train/run*/models/`
- `results/train/run*/logs/`

---

## Render and GIF

### Render with trained model

```bash
cd /path/to/MAPPO
bash scripts/run_render.sh -- --model_dir results/.../train/runN/models --use_render
```

Optional custom render log:

```bash
bash scripts/run_render.sh /path/to/render.log -- --model_dir results/.../train/runN/models --use_render
```

### Save GIF during render

Use these render args:

- `--save_gifs`
- `--render_episodes <num>`
- `--ifi <frame_interval>`

Example:

```bash
cd /path/to/MAPPO
python render.py \
  --model_dir results/.../train/runN/models \
  --use_render \
  --save_gifs \
  --render_episodes 2 \
  --ifi 0.1
```

### Render trajectory summary

```bash
cd /path/to/MAPPO
python scripts/summarize_render_results.py --csv data.csv
```

Optional laplacian-type summary:

```bash
python scripts/summarize_render_results.py --mode laplacian_type2 \
  --laplacian-threshold 0.97 --laplacian-sim-metric cosine01 --sl-threshold 0.97 --csv data.csv
```

### 2 New Components

- `undet_v3_target_latent` pretraining: this module is an offline target-selector pretraining stage. It learns only which target to choose (slot-level logits from relative neighborhood features), and does not train motion control. The pretrained selector head is then loaded into MAPPO actor target-selection branch; with motion_only, the selector can be frozen while training only motion policy.

- Target exchange: exchange is an environment-side heuristic during rollout/training, where nearby agents may swap target_id assignments to reduce team-level assignment cost (e.g., bottleneck/max-distance style objective). It is not part of the offline selector network itself; it is a runtime coordination mechanism on top of target assignment.