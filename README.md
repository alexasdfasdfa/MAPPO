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
