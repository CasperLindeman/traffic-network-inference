# Shared helpers for square-four-to-four turning-outlier diagnostics.

function parse_table_int(value)
    value isa Integer && return Int(value)
    return parse(Int, strip(string(value)))
end

function parse_table_float(value)
    value isa AbstractFloat && return Float64(value)
    return parse(Float64, strip(string(value)))
end

function parse_table_bool(value)
    value isa Bool && return value
    return lowercase(strip(string(value))) == "true"
end

function parse_int_list_string(value)
    text = strip(string(value))
    isempty(text) && return Int[]
    return [parse(Int, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
end

function read_namedtuple_table(input_path)
    data = readdlm(input_path, '\t', String, '\n'; quotes=false)
    size(data, 1) >= 1 || return NamedTuple[]

    header = Tuple(Symbol.(vec(data[1, :])))
    rows = NamedTuple[]

    for row_idx in 2:size(data, 1)
        values = Tuple(data[row_idx, col_idx] for col_idx in 1:size(data, 2))
        push!(rows, NamedTuple{header}(values))
    end

    return rows
end

function posterior_turning_matrices_from_entry_rows(entry_rows)
    Ps = [zeros(Float64, 4, 4) for _ in 1:N_JUNCTIONS]

    for row in entry_rows
        junction = parse_table_int(row.junction)
        incoming_row = parse_table_int(row.incoming_row)
        outgoing_col = parse_table_int(row.outgoing_col)
        Ps[junction][incoming_row, outgoing_col] = parse_table_float(row.posterior_mean)
    end

    for junction in 1:N_JUNCTIONS
        for incoming_row in 1:4
            row_sum = sum(Ps[junction][incoming_row, :])
            @assert row_sum > 0.0 "Missing posterior values for junction $(junction), row $(incoming_row)."
            Ps[junction][incoming_row, :] ./= row_sum
        end
    end

    return turning_matrices(Ps)
end

