# Thesis Experiment Structure

This folder contains the scripts and saved artifacts used by the thesis
numerical experiments.

The E18 corridor experiment is stored in `E18/`. It follows the larger
experiment layout and contains saved outputs from the final ESMDA runs, so its
top-level `run.jl` validates and summarizes artifacts rather than rerunning the
expensive inference.

Small self-contained experiments use:

```text
<experiment>/
|-- inference.jl
|-- plots.jl
|-- run.jl
|-- results.toml
`-- figures/
```

- `inference.jl`: simulator setup, inference call, and numerical summaries.
- `plots.jl`: figure construction from already-computed results or saved
  outputs.
- `run.jl`: main reproducibility entry point for the final thesis artifact.
- `results.toml`: thesis-facing manifest with labels, settings, numerical
  values, source files, outputs, and figure paths.
- `figures/`: final generated figures used by the thesis.

Larger or expensive experiments use:

```text
<experiment>/
|-- run.jl
|-- plots.jl
|-- results.toml
|-- sources/
|-- outputs/
`-- figures/
```

- `sources/`: frozen scripts used for expensive final runs or postprocessing.
- `outputs/`: saved tables, metrics, configs, and intermediate final artifacts.
- `plots.jl`: lightweight plotting/replotting wrappers when figures can be
  regenerated from saved outputs.
- `run.jl`: safe top-level entry point. For expensive experiments this may
  validate artifacts or replot from saved outputs rather than rerun the full
  computational suite.

Shared code lives in family-level `common/` folders. Fixed network
specifications live in `network_specs/`.

For larger experiment families, `common/` may be split by role. For the square
four-to-four experiments:

```text
square_four_to_four/common/
|-- base/
|-- single_scenario/
|-- multi_scenario/
|-- turning_recovery/
`-- map_calibration/
```

- `base/`: network specification, turning-parameter helpers, simulation,
  inference, diagnostics, plotting, and reporting primitives.
- `single_scenario/`: helpers for one square-network scenario.
- `multi_scenario/`: scenario library, multi-scenario datasets, inference,
  metrics, and plots.
- `turning_recovery/`: shared turning-recovery diagnostics used by the
  observation-location, density/noise, and likelihood-weighting experiments.
- `map_calibration/`: optimizer calibration helpers shared by the MAP solver
  calibration entry points.
