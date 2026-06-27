
macro save_carstm_state(filename_sym, vars...)
    """
    BSTM Utility Macro v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Saves a specified set of variables to a JLD2 file. This macro is designed
              to capture the state of a modeling session for later resumption or analysis.
    Inputs:
        - filename_sym: A symbol representing the variable that holds the filename string.
        - vars...: A variable number of symbols representing the variables to save.
    Usage:
        state_filename = "my_model_state.jld2"
        @save_carstm_state(state_filename, data_df, areal_units, m, chn)
    Rationale for v1.2.0:
        - Refactored to accept a variable number of arguments, making it a general-purpose
          state-saving utility instead of being tied to specific variable names.
    """
    return quote
        try
            local fn = $(esc(filename_sym))
            @info "Saving state to $(fn)..."
            # The `JLD2.@save` macro needs the filename as a value and the variables
            # as escaped symbols. The `vars...` are already symbols.
            JLD2.@save fn $([esc(v) for v in vars]...)
            @info "State saved successfully."
        catch e
            @error "Error saving state: $e"
        end
    end
end


macro load_carstm_state(filename_sym, vars...)
    """
    BSTM Utility Macro v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Loads a specified set of variables from a JLD2 file into the current scope.
    Inputs:
        - filename_sym: A symbol representing the variable that holds the filename string.
        - vars...: A variable number of symbols representing the variables to load. If empty,
                   all variables in the file are loaded.
    Usage:
        state_filename = "my_model_state.jld2"
        @load_carstm_state(state_filename, data_df, m)
    Rationale for v1.2.0:
        - Refactored to accept a variable number of arguments for selective loading.
        - If no variables are specified, it loads all variables from the file.
    """
    return quote
        local fn = $(esc(filename_sym))
        if !isfile(fn)
            @error "File $(fn) not found."
        else
            try
                @info "Loading state from $(fn)..."
                # The `JLD2.@load` macro needs the filename as a value.
                # If `vars` is empty, it loads all variables. Otherwise, it loads the specified ones.
                if isempty($(vars))
                    JLD2.@load fn
                else
                    JLD2.@load fn $([esc(v) for v in vars]...)
                end
                @info "State loaded successfully."
            catch e
                @error "Error loading state: $e"
            end
        end
    end
end



function init_params_copy( res=NaN, res0=NaN; load_from_file=false, overrides::Union{Dict, Nothing}=nothing, fn_inits = "init_params.jl2"  )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Copies parameter mean values from a reference MCMC chain summary (`res0`) to a target
              chain summary (`res`), with options to override specific parameters. This is useful for
              initializing a complex model with parameters from a simpler, pre-run model.
    Inputs:
        - res: The target MCMC chain object.
        - res0: The source MCMC chain object.
        - load_from_file: If true, loads parameters from `fn_inits` instead of processing chains.
        - overrides: A dictionary where keys are regex patterns and values are the new values for matching parameters.
        - fn_inits: The filename for saving/loading initial parameters.
    Outputs:
        - A `FillArrays.Fill` object containing the merged mean parameter values, suitable for
          initializing a new MCMC run.
    Rationale for v1.2.0:
        - Replaced the inflexible `override_means` boolean with a flexible `overrides` dictionary,
          allowing programmatic and specific parameter overrides.
    """
  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  ressumm0 = summarize(res0)
  vns0 = ressumm0.nt.parameters
  means0 = ressumm0.nt[2]  # means

  if !isnothing(overrides)
    for (pattern, values) in overrides
        u = findall(x -> occursin(Regex(pattern), String(x)), vns)
        if !isempty(u)
            if length(u) == length(values)
                means[u] .= values
            else
                @warn "Override for '$pattern' failed: length mismatch. Expected $(length(u)), got $(length(values))."
            end
        end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_extract( res=NaN; load_from_file=false, overrides::Union{Dict, Nothing}=nothing, fn_inits = "init_params.jl2"  )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Extracts mean parameter values from a Turing MCMC chain summary to be used as initial
              values for a subsequent run. Includes options for loading from a file or applying
              custom overrides.
    Inputs:
        - res: The MCMC chain object from a previous Turing run.
        - load_from_file: If true, loads parameters directly from `fn_inits`.
        - overrides: A dictionary where keys are regex patterns and values are the new values for matching parameters.
        - fn_inits: The filename for saving/loading initial parameters.
    Outputs:
        - A `FillArrays.Fill` object containing the mean parameter values.
    Rationale for v1.2.0:
        - Replaced `override_means` with a flexible `overrides` dictionary.
    """
  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if !isnothing(overrides)
    for (pattern, values) in overrides
        u = findall(x -> occursin(Regex(pattern), String(x)), vns)
        if !isempty(u)
            if length(u) == length(values)
                means[u] .= values
            else
                @warn "Override for '$pattern' failed: length mismatch. Expected $(length(u)), got $(length(values))."
            end
        end
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_extract(X)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A simplified method to extract parameter names and mean values from a Turing MCMC chain object.
    Inputs:
        - X: The MCMC chain object.
    Outputs:
        - A tuple containing a `FillArrays.Fill` object of the means and a vector of parameter names.
    """
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Rounds a floating-point number `x` to the nearest multiple of `delta`. This is useful
              for discretizing continuous data into regular bins.
    Inputs:
        - x: The input number or vector.
        - delta: The discretization step size.
    Outputs:
        - The discretized number or vector.
    """
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Creates a `DataFrame` from the Cartesian product of named vectors, similar to R's `expand.grid`.
    Inputs:
        - kws: Keyword arguments where each keyword is a symbol for a column name and the value is a vector of values.
    Outputs:
        - A `DataFrame` containing all combinations of the input vectors.
    """
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A simple helper function to print the full contents of a Julia object to the console,
              bypassing truncation that can occur with default display methods.
    """
    show(stdout, "text/plain", x) # display all estimates
end 
 

function modelruntime(o)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Calculates and prints the total runtime of an MCMC sampling process in minutes and
              displays the summary statistics of the resulting chain.
    Inputs:
        - o: The MCMC chain object, which contains timing information in `o.info`.
    """
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A commented-out utility for inspecting the generated code of a function.
              When active, it would use `CodeTracking.@code_string` to print the source code.
    """
end

function firstindexin(a::AbstractArray, b::AbstractArray)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Finds the first occurrence of each element of array `a` within array `b`.
    Inputs:
        - a: The array of elements to search for.
        - b: The array to search within.
    Outputs:
        - An array of the same size as `a`, where each element is the first index of the corresponding
          element from `a` in `b`, or 0 if not found.
    """
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
   

function showtuples(X)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Iterates over the key-value pairs of a `NamedTuple` and prints them to the console,
              rounding numeric values for cleaner display.
    """
    for k in keys(X)
        val = getproperty(X, k)
        # Skip displaying keys with NaN values
        # Check if value is numeric before rounding to avoid errors
        display_val = val isa Number ? round(val, digits=3) : val
        println("$k: $display_val")
    end
end



function showparams(X, keywords=["rho", "phi", "sigma",  "mu_", "l_", "ls_"]; limit=10 )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Filters and displays a summary of parameters from an MCMC chain object based on a list of keywords.
              This is useful for quickly inspecting key hyperparameter posteriors.
    Inputs:
        - X: The MCMC chain object.
        - keywords: A vector of strings to search for within parameter names.
        - limit: The maximum number of matched parameters to display.
    Outputs:
        - A sliced MCMC chain object containing only the matched parameters.
    """
    # Create a regex pattern by joining keywords with the pipe '|' operator
    pattern = Regex(join(keywords, "|"))

    # Filter the parameter list
    matched_params = filter(p -> occursin(pattern, string(p)), FlexiChains.parameters(X))

    # Display the filtered slice
    if isempty(matched_params)
        println("No parameters matched keywords: $keywords")
    else
        out = X[matched_params[1:min(limit, end)]]
        # display(out)
        return out
    end
end





function random_correlation_matrix(d=3, eta=1)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Generates a random correlation matrix of a given dimension using the "Onion Method",
              which is related to the LKJ distribution.
    Inputs:
        - d: The dimension of the square correlation matrix.
        - eta: A parameter controlling the distribution of correlations. `eta=1` corresponds to a
               uniform distribution over correlation matrices. Larger values of `eta` push the
               matrix closer to the identity matrix.
    Outputs:
        - A `d x d` random correlation matrix.
    Reference:
        - https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
    Rationale for v1.0.1:
        - Replaced the full eigendecomposition in the loop with a more efficient Cholesky decomposition
          for computing the matrix square root, which aligns with the canonical "Onion Method" algorithm.
        - Corrected the calculation of the vector `q` to use standard matrix multiplication.
    """
    beta = eta + (d - 2) / 2
    u = rand(Beta(beta, beta))
    r12 = 2 * u - 1
    S = [1 r12; r12 1]

    for k = 3:d
        beta -= 0.5
        y = rand(Beta((k - 1) / 2, beta))
        r = sqrt(y)
        theta = randn(k - 1)
        theta /= norm(theta)
        w = r * theta

        # Use Cholesky decomposition for the matrix square root, which is more efficient.
        # The algorithm requires a matrix R such that S = R'R. The upper Cholesky factor C.U satisfies this.
        # Then, q = R'w, which is equivalent to C.L * w.
        C = cholesky(Symmetric(S))
        q = C.L * w

        S = [S q; q' 1]
    end
    return S
end





function turingindex( indices, sym=nothing, dims=nothing  ) 
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A helper function to extract parameter indices from a Turing model's internal
              variable information structure.
    Inputs:
        - indices: The `VarInfo` metadata from a Turing model, or the model itself.
        - sym: The symbol of the parameter to extract indices for. If `nothing`, enumerates all keys.
               If `"varnames"`, returns all variable names.
        - dims: Optional dimensions to reshape the output index array.
    Outputs:
        - A vector or array of indices corresponding to the specified parameter.
    Rationale for v1.2.0:
        - Added a `haskey` check to provide a more informative error when a symbol is not found,
          preventing a `KeyError`.
    """
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      if !haskey(indices, sym)
          error("Symbol ':$sym' not found in model variable information. Available keys: $(keys(indices))")
      end
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
end


 
function dataframe_to_named_array(df::DataFrame)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Converts a `DataFrame` into a `NamedArray` for use in internal model processing,
              preserving column names as the second dimension's names.
    """
    mat = Matrix(df)
    return NamedArray(mat, (1:size(mat, 1), Symbol.(names(df))))
end
