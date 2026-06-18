# TrafficNetworks

Julia package and thesis experiment code for macroscopic traffic-network
simulation and inference.

## Contents

- `src/`: package source code.
- `thesis_experiments/`: final experiment scripts, saved outputs, figures, and
  result manifests used in the thesis.
- `Project.toml` and `Manifest.toml`: Julia environment for reproducing the
  package and experiment code.
- `test/`: lightweight package and workflow checks.

Exploratory notebooks and older generated scratch outputs are not part of the
final thesis-ready package tree.

## Quick checks

From the repository root:

```powershell
julia --project=TrafficNetworks -e "using TrafficNetworks; println(:ok)"
```

Run the package checks with:

```powershell
julia --project=TrafficNetworks TrafficNetworks/test/runtests.jl
```

Run individual thesis experiment entry points from
`TrafficNetworks/thesis_experiments/`. Expensive experiments keep saved final
outputs and may validate or replot instead of rerunning inference.
