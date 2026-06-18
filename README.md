# Master thesis repository

This repository contains the final master thesis PDF, the Julia code, and the
saved experiment artifacts used for the numerical experiments.

## Layout

- `Master_Thesis.pdf`: final compiled master thesis PDF.
- `Semester_Project.pdf`: PDF of the preceding semester project report.
- `TrafficNetworks/`: Julia package for macroscopic traffic-network
  simulations, inference utilities, and plotting helpers.
- `TrafficNetworks/thesis_experiments/`: final thesis experiment entry points,
  saved outputs, generated figures, and result manifests.
- `TrafficNetworks/thesis_experiments/network_specs/`: fixed network
  specifications used by the small synthetic experiments.

Exploratory notebooks, local editor state, template work, build artifacts, and
older scratch files have been moved out of this final repository folder.

## Julia package and experiments

Instantiate the Julia environment before running tests or experiment scripts:

```powershell
julia --project=TrafficNetworks -e "using Pkg; Pkg.instantiate()"
```

The final thesis experiments are organized under
`TrafficNetworks/thesis_experiments/`. Most experiments expose a top-level
`run.jl`; expensive experiments validate or replot from saved outputs rather
than rerunning the full computational suite.

Example:

```powershell
julia --project=TrafficNetworks TrafficNetworks/thesis_experiments/two_to_two/run.jl
```

## License

This repository is released under the MIT license; see `LICENSE`.
