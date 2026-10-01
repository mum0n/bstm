function parse_errors(ex, src::Vector{String})
    ex isa Expr || return String[]
    out = String[]
    if ex.head in (:error, :incomplete)
        line = try
            Int(ex.line)
        catch
            0
        end
        detail = ex.args[1]
        # A ParseError embeds the ENTIRE file text, so truncate to the first lines.
        msg = try
            string(detail)
        catch
            repr(detail)
        end
        msg = join(first(split(msg, '\n'), 3), "\n      ")
        ctx = 0 < line <= length(src) ? "\n      line $line: " * src[line] : ""
        push!(out, "line $line: $msg$ctx")
    end
    for a in ex.args
        append!(out, parse_errors(a, src))
    end
    return out
end

for f in ("src/reconstruction.jl", "src/likelihoods.jl", "src/model.jl", "src/par.jl",
          "src/pipeline.jl", "src/input_output.jl", "src/bstm.jl",
          "src/components/fitc.jl", "src/components/ar1.jl",
          "test/test_reconstruction.jl", "test/test_par.jl", "test/test_core_formula.jl",
          "test/test_likelihoods.jl")
    text = read(f, String)
    src = collect(eachline(IOBuffer(text)))
    try
        ex = Meta.parseall(text)
        errs = parse_errors(ex, src)
        if isempty(errs)
            println("PARSE OK    $f")
        else
            println("PARSE $(length(errs)) ERR $f")
            for e in errs
                println("    $e")
            end
        end
    catch e
        println("PARSE THREW  $f")
        println("  ", sprint(showerror, e))
    end
end

