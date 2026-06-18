# E18 Corridor Experiment

This folder contains the final saved artifacts for the E18 corridor numerical
experiment.

The expensive ESMDA and L-BFGS runs are already saved in `outputs/`. The
top-level `run.jl` validates the saved files and prints the main numerical
summaries; it does not rerun the inference. The `plots.jl` wrapper only
synchronizes figures that were already generated from saved outputs.

```text
E18/
|-- run.jl
|-- plots.jl
|-- results.toml
|-- sources/
|-- outputs/
`-- figures/
```

- `sources/source_json/`: original downloaded E18 API JSON files.
- `sources/`: frozen scripts used to build the graph, run inference, and make
  diagnostics.
- `outputs/graph/`: final pruned graph and simulator network files.
- `outputs/selection/`: chosen inference targets and sensor locations.
- `outputs/inference_64x2_180s/` and `outputs/inference_128x4_180s/`: saved
  ESMDA tables and diagnostics used in the thesis.
- `outputs/lbfgs_checkpoint_backtracking_alpha0050_retry_180s/`: final L-BFGS
  MAP output used in the thesis.
- `outputs/lbfgs_prior_mean_backtracking_180s/`: retained start checkpoint for
  the final L-BFGS continuation.
- `figures/`: final thesis-facing figures.
