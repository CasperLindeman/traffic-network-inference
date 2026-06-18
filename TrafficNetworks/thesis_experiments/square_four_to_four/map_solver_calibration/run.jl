import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using TOML

const MAP_CALIBRATION_ARTIFACT_EXTENSIONS = (".tsv", ".txt")

function collect_manifest_artifact_paths!(paths::Vector{String}, value)
    if value isa AbstractDict
        for child in values(value)
            collect_manifest_artifact_paths!(paths, child)
        end
    elseif value isa AbstractVector
        for child in value
            collect_manifest_artifact_paths!(paths, child)
        end
    elseif value isa AbstractString && any(endswith(value, ext) for ext in MAP_CALIBRATION_ARTIFACT_EXTENSIONS)
        push!(paths, value)
    end

    return paths
end

function check_map_solver_calibration_artifacts(; manifest_path=joinpath(@__DIR__, "results.toml"))
    manifest = TOML.parsefile(manifest_path)
    relative_paths = unique!(sort!(collect_manifest_artifact_paths!(String[], manifest)))
    missing_paths = [path for path in relative_paths if !isfile(joinpath(@__DIR__, path))]

    println("MAP solver calibration artifact check")
    println("-------------------------------------")
    println("manifest: ", manifest_path)
    println("referenced artifacts: ", length(relative_paths))
    println("missing artifacts: ", length(missing_paths))
    for path in missing_paths
        println("  missing: ", path)
    end

    isempty(missing_paths) || error("Missing map-solver calibration artifacts.")

    return (
        manifest_path=manifest_path,
        artifact_paths=relative_paths,
    )
end

check_map_solver_calibration_artifacts()
