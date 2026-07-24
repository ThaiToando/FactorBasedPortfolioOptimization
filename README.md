
# fbpo: Factor-Based Portfolio Optimization

Independent reproduction and extension of Auh, J.K. & Cho, W. (2023),
"Factor-based portfolio optimization," *Economics Letters* 228, 111137.

Research and educational purposes only.

## Reproduce it

```bash
uv sync --all-extras
uv run fbpo config-show --config configs/base.yaml
uv run pytest -q
```
