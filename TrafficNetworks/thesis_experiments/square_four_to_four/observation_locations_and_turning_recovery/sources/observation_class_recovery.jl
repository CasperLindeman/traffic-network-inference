ENV["TRAFFICNETWORKS_SKIP_SQUARE_FOUR_TO_FOUR_TURNING_OUTLIER_STRUCTURE"] = "1"

include(joinpath(@__DIR__, "turning_recovery_helpers.jl"))

using Random
using Statistics
using Printf
using Distributions
using LaTeXStrings

const EXP_A_CLASS_ORDER = [
    ("both", "source+target observed"),
    ("target", "target observed"),
    ("source", "source observed"),
    ("neither", "neither directly observed"),
]

function exp_a_rank_values(values::AbstractVector{<:Real})
    order = sortperm(collect(values))
    ranks = zeros(Float64, length(values))
    pos = 1

    while pos <= length(order)
        stop = pos
        while stop < length(order) && values[order[stop + 1]] == values[order[pos]]
            stop += 1
        end

        rank_value = (pos + stop) / 2
        for idx in pos:stop
            ranks[order[idx]] = rank_value
        end
        pos = stop + 1
    end

    return ranks
end

function friedman_test(matrix::AbstractMatrix{<:Real})
    n, k = size(matrix)
    rank_sums = zeros(Float64, k)

    for row in 1:n
        rank_sums .+= exp_a_rank_values(view(matrix, row, :))
    end

    statistic = 12.0 / (n * k * (k + 1)) * sum(abs2, rank_sums) - 3.0 * n * (k + 1)
    p_value = ccdf(Chisq(k - 1), statistic)
    return (statistic=statistic, p_value=p_value, rank_sums=rank_sums)
end

function signed_rank_exact_pvalue(differences::AbstractVector{<:Real})
    nonzero = [Float64(value) for value in differences if value != 0.0]
    n = length(nonzero)
    n == 0 && return 1.0

    ranks = exp_a_rank_values(abs.(nonzero))
    observed = sum(ranks[nonzero .> 0.0])
    scale = all(isapprox.(ranks, round.(ranks); atol=1e-12)) ? 1 : 2
    scaled_ranks = round.(Int, scale .* ranks)
    scaled_observed = round(Int, scale * observed)
    max_sum = sum(scaled_ranks)
    counts = zeros(Int, max_sum + 1)
    counts[1] = 1

    for rank in scaled_ranks
        for value in max_sum:-1:rank
            counts[value + 1] += counts[value - rank + 1]
        end
    end

    return sum(counts[(scaled_observed + 1):end]) / 2.0^n
end

function signed_rank_normal_pvalue(differences::AbstractVector{<:Real})
    nonzero = [Float64(value) for value in differences if value != 0.0]
    n = length(nonzero)
    n == 0 && return 1.0

    ranks = exp_a_rank_values(abs.(nonzero))
    observed = sum(ranks[nonzero .> 0.0])
    mean_rank = n * (n + 1) / 4
    var_rank = n * (n + 1) * (2n + 1) / 24
    z = (observed - mean_rank - 0.5) / sqrt(var_rank)
    return ccdf(Normal(), z)
end

function signed_rank_greater_pvalue(differences::AbstractVector{<:Real})
    n_nonzero = count(!=(0.0), differences)
    n_nonzero <= 50 && return signed_rank_exact_pvalue(differences)
    return signed_rank_normal_pvalue(differences)
end

function signed_rank_greater_stats(differences::AbstractVector{<:Real})
    nonzero = [Float64(value) for value in differences if value != 0.0]
    n = length(nonzero)
    if n == 0
        return (
            n_nonzero=0,
            w_plus=0.0,
            w_minus=0.0,
            z_cc=NaN,
            r_z=NaN,
            rank_biserial=NaN,
            p_value=1.0,
        )
    end

    ranks = exp_a_rank_values(abs.(nonzero))
    w_plus = sum(ranks[nonzero .> 0.0])
    w_minus = sum(ranks[nonzero .< 0.0])
    mean_rank = n * (n + 1) / 4
    var_rank = n * (n + 1) * (2n + 1) / 24
    z_cc = (w_plus - mean_rank - 0.5) / sqrt(var_rank)
    rank_total = n * (n + 1) / 2

    return (
        n_nonzero=n,
        w_plus=w_plus,
        w_minus=w_minus,
        z_cc=z_cc,
        r_z=z_cc / sqrt(n),
        rank_biserial=(w_plus - w_minus) / rank_total,
        p_value=signed_rank_greater_pvalue(differences),
    )
end

function holm_adjust(p_values::AbstractVector{<:Real})
    m = length(p_values)
    order = sortperm(collect(p_values))
    adjusted = zeros(Float64, m)
    running_max = 0.0

    for (rank_idx, original_idx) in enumerate(order)
        value = min(1.0, (m - rank_idx + 1) * Float64(p_values[original_idx]))
        running_max = max(running_max, value)
        adjusted[original_idx] = running_max
    end

    return adjusted
end

function bootstrap_ci(values::AbstractVector{<:Real}; n_bootstrap=10_000, seed=73_001)
    rng = MersenneTwister(seed)
    n = length(values)
    means = Vector{Float64}(undef, n_bootstrap)

    for iter in 1:n_bootstrap
        total = 0.0
        for _ in 1:n
            total += values[rand(rng, 1:n)]
        end
        means[iter] = total / n
    end

    return (lower=quantile(means, 0.025), upper=quantile(means, 0.975))
end

function class_lookup_from_metadata(metadata)
    lookup = Dict{String, Vector{Int}}()
    for (short_label, long_label) in EXP_A_CLASS_ORDER
        lookup[short_label] = [row.global_entry for row in metadata if row.direct_observation_class == long_label]
    end
    return lookup
end

function write_exp_a_rmse_plot(class_rows, seed_rows, output_path)
    class_labels = first.(EXP_A_CLASS_ORDER)
    means = [only([row.mean_rmse for row in class_rows if row.class == class_label]) for class_label in class_labels]
    lowers = [only([row.min_rmse for row in class_rows if row.class == class_label]) for class_label in class_labels]
    uppers = [only([row.max_rmse for row in class_rows if row.class == class_label]) for class_label in class_labels]
    x = collect(1:length(class_labels))

    plt = plot(
        legend=false,
        ylabel=L"\mathrm{Class\ wise\ RMSE}_{\theta}",
        xlabel="Observation class",
        xticks=(x, class_labels),
        xlims=(0.5, length(class_labels) + 0.5),
        ylims=(0.0, 1.12 * maximum(uppers)),
        size=(760, 420),
        dpi=220,
        left_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        guidefontsize=15,
        tickfontsize=13,
        legendfontsize=12,
        titlefontsize=18,
        grid=true,
        framestyle=:axes,
        foreground_color_border=:black,
        foreground_color_axis=:black,
    )

    cap = 0.035
    for (idx, (mean_value, lower_value, upper_value)) in enumerate(zip(means, lowers, uppers))
        plot!(plt, [idx, idx], [lower_value, upper_value]; color=:black, alpha=0.86, linewidth=2.0, label="")
        plot!(plt, [idx - cap, idx + cap], [lower_value, lower_value]; color=:black, alpha=0.86, linewidth=2.0, label="")
        plot!(plt, [idx - cap, idx + cap], [upper_value, upper_value]; color=:black, alpha=0.86, linewidth=2.0, label="")
        scatter!(
            plt,
            [idx],
            [mean_value];
            color=:steelblue4,
            markerstrokecolor=:black,
            markerstrokewidth=1.1,
            markersize=7,
            label="",
        )
    end

    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function run_exp_a_fixed_layout(;
    n_seeds=30,
    noise_seeds=collect(1:n_seeds),
    esmda_seed=1,
    regime=MultiScenarioDataRegime("multi_scenario_12x", 1, 12),
    ensemble_size=192,
    esmda_maxiters=6,
    floor_fraction=0.15,
    output_dir=TURNING_OUTLIER_OUTPUT_DIR,
)
    mkpath(output_dir)

    P_true = true_turning_matrices()
    dataset = build_multi_scenario_dataset(regime; peak_noise_sigma=MULTI_SCENARIO_PEAK_NOISE_SIGMA)
    metadata = turning_entry_metadata(first(dataset.setups))
    class_lookup = class_lookup_from_metadata(metadata)

    for (class_label, indices) in class_lookup
        isempty(indices) && error("Class $(class_label) has no entries; cannot run four-class Exp A.")
    end

    config_path = write_config_file(
        joinpath(output_dir, "exp_a_config.txt"),
        [
            "experiment = Exp A fixed layout entry-level observation class test",
            "noise_seeds = $(join(noise_seeds, ","))",
            "esmda_seed = $(esmda_seed)",
            "regime = $(regime.label)",
            "scenario_count = $(regime.scenario_count)",
            "horizon_factor = $(regime.horizon_factor)",
            "observation_count = $(dataset_observation_length(dataset))",
            "observation_multiplier = $(@sprintf("%.2f", observation_multiplier(dataset)))",
            "ensemble_size = $(ensemble_size)",
            "esmda_maxiters = $(esmda_maxiters)",
            "floor_fraction = $(floor_fraction)",
            "class_order = $(join(first.(EXP_A_CLASS_ORDER), ","))",
        ],
    )

    seed_rows_path = joinpath(output_dir, "exp_a_seed_class_rmse.tsv")
    seed_abs_rows_path = joinpath(output_dir, "exp_a_seed_class_mean_abs_errors.tsv")
    entry_rows_path = joinpath(output_dir, "exp_a_entry_errors.tsv")
    seed_rows = isfile(seed_rows_path) ? read_namedtuple_table(seed_rows_path) : NamedTuple[]
    seed_abs_rows = isfile(seed_abs_rows_path) ? read_namedtuple_table(seed_abs_rows_path) : NamedTuple[]
    entry_rows = isfile(entry_rows_path) ? read_namedtuple_table(entry_rows_path) : NamedTuple[]

    completed_seeds = Set{Int}()
    for noise_seed in noise_seeds
        class_count = count(row -> parse_table_int(row.noise_seed) == noise_seed, seed_rows)
        class_count == length(EXP_A_CLASS_ORDER) && push!(completed_seeds, noise_seed)
    end

    if !isempty(completed_seeds)
        println("Reusing completed seeds: ", join(sort(collect(completed_seeds)), ","))
    end

    start_time = time()
    fitted_count = 0

    for (seed_pos, noise_seed) in enumerate(noise_seeds)
        if noise_seed in completed_seeds
            @printf("seed %d/%d noise_seed=%d already complete; skipping fit\n", seed_pos, length(noise_seeds), noise_seed)
            flush(stdout)
            continue
        end

        observations = generate_physical_dataset_observations(P_true, dataset; seed=noise_seed, floor_fraction=floor_fraction)
        fit_seconds = @elapsed begin
            esmda = run_esmda_multi_scenario(
                dataset,
                observations.y_obs,
                observations.sigma_model;
                seed=esmda_seed,
                ensemble_size=ensemble_size,
                esmda_maxiters=esmda_maxiters,
                P_true=P_true,
            )
        end

        abs_errors = abs.(esmda.entry_post_mean .- esmda.entry_true)
        for meta in metadata
            push!(
                entry_rows,
                (
                    noise_seed=noise_seed,
                    esmda_seed=esmda_seed,
                    global_entry=meta.global_entry,
                    entry_label=meta.entry_label,
                    junction=meta.junction,
                    incoming_row=meta.incoming_row,
                    outgoing_col=meta.outgoing_col,
                    source_road=meta.source_road,
                    target_road=meta.target_road,
                    class=first(pair[1] for pair in EXP_A_CLASS_ORDER if pair[2] == meta.direct_observation_class),
                    direct_observation_class=meta.direct_observation_class,
                    truth=esmda.entry_true[meta.global_entry],
                    posterior_mean=esmda.entry_post_mean[meta.global_entry],
                    abs_error=abs_errors[meta.global_entry],
                ),
            )
        end

        for (class_label, _) in EXP_A_CLASS_ORDER
            indices = class_lookup[class_label]
            push!(
                seed_rows,
                (
                    noise_seed=noise_seed,
                    class=class_label,
                    entry_count=length(indices),
                    turning_rmse=sqrt(mean(abs2, abs_errors[indices])),
                ),
            )
            push!(
                seed_abs_rows,
                (
                    noise_seed=noise_seed,
                    esmda_seed=esmda_seed,
                    class=class_label,
                    entry_count=length(indices),
                    mean_abs_error=mean(abs_errors[indices]),
                    median_abs_error=median(abs_errors[indices]),
                    max_abs_error=maximum(abs_errors[indices]),
                    fit_seconds=fit_seconds,
                ),
            )
        end

        write_namedtuple_table(seed_rows, seed_rows_path)
        write_namedtuple_table(seed_abs_rows, seed_abs_rows_path)
        write_namedtuple_table(entry_rows, entry_rows_path)

        fitted_count += 1
        elapsed = time() - start_time
        mean_seed_seconds = elapsed / fitted_count
        remaining_fits = length([seed for seed in noise_seeds[seed_pos:end] if !(seed in completed_seeds)]) - 1
        remaining_seconds = mean_seed_seconds * max(remaining_fits, 0)
        @printf(
            "seed %d/%d noise_seed=%d fit %.1fs elapsed %.1fmin eta %.1fmin\n",
            seed_pos,
            length(noise_seeds),
            noise_seed,
            fit_seconds,
            elapsed / 60,
            remaining_seconds / 60,
        )
        flush(stdout)
    end

    class_labels = first.(EXP_A_CLASS_ORDER)
    available_seeds = sort([seed for seed in noise_seeds if count(row -> parse_table_int(row.noise_seed) == seed, seed_rows) == length(class_labels)])
    matrix = Matrix{Float64}(undef, length(available_seeds), length(class_labels))
    for (row_idx, noise_seed) in enumerate(available_seeds)
        for (col_idx, class_label) in enumerate(class_labels)
            matrix[row_idx, col_idx] = only([
                parse_table_float(row.turning_rmse) for row in seed_rows
                if parse_table_int(row.noise_seed) == noise_seed && string(row.class) == class_label
            ])
        end
    end

    friedman = friedman_test(matrix)
    contrast_specs = [
        ("neither_vs_both", "neither", "both", "planned"),
        ("source_vs_both", "source", "both", "planned"),
        ("target_vs_both", "target", "both", "planned"),
        ("source_vs_target", "source", "target", "exploratory"),
        ("neither_vs_target", "neither", "target", "exploratory"),
        ("neither_vs_source", "neither", "source", "exploratory"),
    ]
    raw_p = Float64[]
    contrast_rows = NamedTuple[]

    for (contrast_label, greater_class, lower_class, contrast_type) in contrast_specs
        greater_idx = findfirst(==(greater_class), class_labels)
        lower_idx = findfirst(==(lower_class), class_labels)
        differences = matrix[:, greater_idx] .- matrix[:, lower_idx]
        stats = signed_rank_greater_stats(differences)
        push!(raw_p, stats.p_value)
        push!(
            contrast_rows,
            (
                contrast=contrast_label,
                alternative="$(greater_class) > $(lower_class)",
                type=contrast_type,
                mean_difference=mean(differences),
                median_difference=median(differences),
                w_plus=stats.w_plus,
                raw_p=stats.p_value,
                holm_p=NaN,
            ),
        )
    end

    adjusted_p = holm_adjust(raw_p)
    contrast_rows = [
        merge(row, (holm_p=adjusted_p[idx],)) for (idx, row) in enumerate(contrast_rows)
    ]

    class_summary_rows = NamedTuple[]
    for class_label in class_labels
        values = [parse_table_float(row.turning_rmse) for row in seed_rows if string(row.class) == class_label]
        push!(
            class_summary_rows,
            (
                class=class_label,
                entry_count=only(unique([parse_table_int(row.entry_count) for row in seed_rows if string(row.class) == class_label])),
                seed_count=length(values),
                mean_rmse=mean(values),
                seed_sd=std(values),
                min_rmse=minimum(values),
                max_rmse=maximum(values),
            ),
        )
    end

    analysis_rows = [
        (
            test="friedman",
            statistic=friedman.statistic,
            p_value=friedman.p_value,
            n_seeds=length(available_seeds),
            n_classes=length(class_labels),
        ),
    ]

    write_namedtuple_table(class_summary_rows, joinpath(output_dir, "exp_a_class_rmse_summary.tsv"))
    write_namedtuple_table(analysis_rows, joinpath(output_dir, "exp_a_friedman_rmse.tsv"))
    write_namedtuple_table(contrast_rows, joinpath(output_dir, "exp_a_pairwise_rmse_tests.tsv"))
    figure_dir = normpath(joinpath(@__DIR__, "..", "figures"))
    figure_path = write_exp_a_rmse_plot(
        class_summary_rows,
        seed_rows,
        joinpath(figure_dir, "square_observation_locations_turning_error.png"),
    )

    println()
    println("Exp A fixed-layout observation-class experiment complete")
    println("-------------------------------------------------------")
    println(config_path)
    println(seed_rows_path)
    println(seed_abs_rows_path)
    println(joinpath(output_dir, "exp_a_entry_errors.tsv"))
    println(joinpath(output_dir, "exp_a_class_rmse_summary.tsv"))
    println(joinpath(output_dir, "exp_a_friedman_rmse.tsv"))
    println(joinpath(output_dir, "exp_a_pairwise_rmse_tests.tsv"))
    println(figure_path)
    println()
    @printf("Friedman statistic %.4f p %.6g\n", friedman.statistic, friedman.p_value)
    for row in contrast_rows
        @printf("%s mean_diff %.6f raw_p %.6g holm_p %.6g\n", row.contrast, row.mean_difference, row.raw_p, row.holm_p)
    end

    return (
        output_dir=output_dir,
        seed_rows=seed_rows,
        entry_rows=entry_rows,
        class_summary_rows=class_summary_rows,
        analysis_rows=analysis_rows,
        contrast_rows=contrast_rows,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    n_seeds = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : parse(Int, get(ENV, "EXP_A_N_SEEDS", "30"))
    output_dir = get(ENV, "EXP_A_OUTPUT_DIR", TURNING_OUTLIER_OUTPUT_DIR)
    esmda_seed = parse(Int, get(ENV, "EXP_A_ESMDA_SEED", "1"))
    run_exp_a_fixed_layout(; n_seeds=n_seeds, esmda_seed=esmda_seed, output_dir=output_dir)
end
