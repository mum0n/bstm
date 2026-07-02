using Plots, StatsPlots, DataFrames, Statistics, MCMCChains, FlexiChains, Distributions, DynamicPPL, SparseArrays, LinearAlgebra, StatsBase


function plot_choropleth(values::AbstractVector, polygons; title::String="", clims=nothing)
    """
    Generates a choropleth map from a vector of values and corresponding polygons.
    This is a helper function for visualizing spatial effects.
    """
    if length(values) != length(polygons)
        error("The number of values must match the number of polygons.")
    end

    if isnothing(clims)
        clims = (minimum(values), maximum(values))
    end

    p = plot(
        title=title,
        aspect_ratio=:equal,
        legend=false,
        framestyle=:none
    )

    plot!(p, polygons, fill_z=values, color=:viridis, colorbar_title="Value", clims=clims)

    return p
end


function summarize_array(samples::AbstractArray; alpha=0.05)
    # Computes summary statistics for posterior samples.
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end

    dims = size(samples)
    sample_dim = length(dims)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    low_bound = dropdims(mapslices(x -> Statistics.quantile(x, low_prob), samples, dims=sample_dim), dims=sample_dim)
    high_bound = dropdims(mapslices(x -> Statistics.quantile(x, high_prob), samples, dims=sample_dim), dims=sample_dim)

    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end


# --- Reconstruction Engine (from reconstruction_engine.jl) ---

include("reconstruction_engine.jl")


# --- Plot Generation ---

function _generate_plots(res, M; au=nothing, data=nothing, ts=1, outcome=1)
    # This function adapts the logic from the original `model_results_plots`
    # to generate a dictionary of plots.

    plots = Dict{Symbol, Any}()
    pstats = res # The 'res' object passed here is the direct output of _reconstruct.
    y_obs = isnothing(data) ? M.y_obs : data.y # y_obs is in M, not in the reconstruction result.
    polygons = isnothing(au) ? nothing : get(au, :polygons, nothing)
    centroids = isnothing(au) ? nothing : get(au, :centroids, nothing)

    # Determine architecture for multivariate plotting logic
    arch_type = if get(M, :model_arch, "univariate") == "multivariate"
        MultivariateArchitecture()
    else
        UnivariateArchitecture()
    end

    # Panel 1: Posterior Predictive Check (PPC)
    is_mv = (y_obs isa AbstractMatrix && size(y_obs, 2) > 1)
    y_p = is_mv ? pstats.predictions_denoised.mean[:, outcome] : vec(pstats.predictions_denoised.mean)
    y_o = is_mv ? y_obs[:, outcome] : vec(y_obs)

    if length(y_p) == length(y_o)
        p_ppc = scatter(vec(y_p), vec(y_o), title="Posterior Predictive Check", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
        clean_p = filter(!isnan, y_p)
        clean_o = filter(!isnan, y_o)
        if !isempty(clean_p) && !isempty(clean_o)
            min_val = min(minimum(clean_p), minimum(clean_o))
            max_val = max(maximum(clean_p), maximum(clean_o))
            plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5)
        end
        plots[:ppc] = p_ppc
    end

    # Helper for spatial plots
    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if !isnothing(field_data) && hasproperty(field_data, :mean)
            s_mean = vec(collect(field_data.mean))
            if !all(iszero, s_mean) && (!isnothing(polygons) || !isnothing(centroids))
                if !isnothing(polygons) && length(polygons) >= length(s_mean)
                    return plot_choropleth(s_mean, polygons; title=title_str)
                elseif !isnothing(centroids)
                    p_map = scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal)
                    return p_map
                end
            end
        end
        return nothing
    end

    # Spatial Plots
    s_field_denoised = (arch_type isa MultivariateArchitecture) ? get(pstats, :spatial_denoised, [nothing])[outcome] : get(pstats, :spatial_denoised, nothing)
    p_spatial_denoised = _create_choropleth_plot(s_field_denoised, "Spatial Denoised Effect", polygons, centroids)
    !isnothing(p_spatial_denoised) && (plots[:spatial_denoised] = p_spatial_denoised)

    s_field_noisy = (arch_type isa MultivariateArchitecture) ? get(pstats, :spatial_noisy, [nothing])[outcome] : get(pstats, :spatial_noisy, nothing)
    p_spatial_noisy = _create_choropleth_plot(s_field_noisy, "Total Spatial Effect", polygons, centroids)
    !isnothing(p_spatial_noisy) && (plots[:spatial_noisy] = p_spatial_noisy)

    # Temporal Plot
    if hasproperty(pstats, :temporal)
        raw_t = pstats.temporal
        t_field = (arch_type isa MultivariateArchitecture) ? raw_t[outcome] : raw_t
        if !isnothing(t_field) && hasproperty(t_field, :mean) && !all(iszero, t_field.mean)
            tm, tl, tu = vec(t_field.mean), vec(t_field.lower), vec(t_field.upper)
            plots[:temporal] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index")
        end
    end

    # Seasonal Plot
    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal) && hasproperty(pstats.seasonal, :mean) && !all(iszero, pstats.seasonal.mean)
        um, ul, uu = vec(pstats.seasonal.mean), vec(pstats.seasonal.lower), vec(pstats.seasonal.upper)
        plots[:seasonal] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Cycle", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Season Bin")
    end

    # Smoothed Covariate Effects
    if hasproperty(pstats, :smooth_effects) && pstats.smooth_effects isa Dict
        smooth_plots = Dict()
        for (var_sym, smooth_summary) in pstats.smooth_effects
            if !isnothing(data) && hasproperty(data, var_sym)
                covariate_data = data[!, var_sym]
                p_order = sortperm(covariate_data)
                sm, sl, su = vec(smooth_summary.mean), vec(smooth_summary.lower), vec(smooth_summary.upper)
                smooth_plots[var_sym] = plot(covariate_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]),
                                             title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Effect on eta",
                                             legend=false, color=:darkorange, fillalpha=0.2)
            end
        end
        if !isempty(smooth_plots); plots[:smooth_effects] = smooth_plots; end
    end

    # Fixed Effects
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects) && hasproperty(pstats.fixed_effects, :mean)
        fm, fl, fu = vec(pstats.fixed_effects.mean), vec(pstats.fixed_effects.lower), vec(pstats.fixed_effects.upper)
        n_coeffs = length(fm)
        if n_coeffs > 0
            p_forest = scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), title="Fixed Effects", xlabel="Coefficient", ylabel="Index", markersize=4, color=:black, legend=false, yticks=(1:n_coeffs, ["β$i" for i in 1:n_coeffs]))
            vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
            plots[:fixed_effects] = p_forest
        end
    end

    return NamedTuple(plots)
end


# --- Main User-Facing Functions ---

function model_results_comprehensive(model, chain; au=nothing, data=nothing, n_samples=1000, alpha=0.05)
    println("--- Starting Comprehensive Model Reporting ---")

    # Metadata and Architecture Extraction
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")
    model_family = get(M, :model_family, "gaussian")

    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # Latent Manifold Reconstruction
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # Performance Metric Assessment
    y_pred = res.predictions_denoised.mean
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0

    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]
        rmse_val = sqrt(Statistics.mean((obs_v .- pred_v).^2))
        try
            r_pearson = Statistics.cor(obs_v, pred_v)
        catch
            r_pearson = 0.0
        end
    end
  # The `get` method with a default value is not implemented for the `FlexiChainMetadata` type.
    # This logic manually checks for the property's existence and provides a default,
    # preserving the original code's assumption that the value is a `Ref` that needs dereferencing.
    sampling_time = if hasproperty(chain, :_metadata)
        time_val_ref = if hasproperty(chain._metadata, :sampling_time)
            getproperty(chain._metadata, :sampling_time)
        else 
            Ref(0.0) 
        end
        # The value is expected to be a Ref, so we dereference it.
        time_val_ref[]
    else 
        0.0 
    end

    sum_stats_df = DataFrame(MCMCChains.summarystats(chain))
    
    mean_rhat = 1.0
    min_ess = 0.0

    try
        rhat_vector = sum_stats_df[!, :rhat]
        ess_bulk_vector = sum_stats_df[!, :ess_bulk]
        mean_rhat = Statistics.mean(filter(!isnan, rhat_vector))
        min_ess = Statistics.minimum(filter(!isnan, ess_bulk_vector))
    catch
        try
            ess_alt = sum_stats_df[!, :ess]
            min_ess = Statistics.minimum(filter(!isnan, ess_alt))
        catch
            mean_rhat = 1.0
            min_ess = 0.0
        end
    end

    waic_val = get(res, :waic, 0.0)
    ess_rate = sampling_time > 0 ? round(min_ess/sampling_time, digits=2) : 0.0

    # Generate Plots
    plots = _generate_plots(res, M; au=au, data=data)

    # Final Report
    println("\n--- Model Metadata ---")
    println("Architecture:     ", raw_arch)
    println("Family:           ", model_family)
    println("Space Component:  ", get(M, :model_space, "none"))
    println("Time Component:   ", get(M, :model_time, "none"))
    println("Seasonal Component: ", get(M, :model_season, "none"))

    println("\n--- Performance Metrics ---")
    println("Compute Time:     ", round(sampling_time, digits=2), " seconds")
    println("RMSE:             ", round(rmse_val, digits=4))
    println("Pearson r:        ", round(r_pearson, digits=4))
    println("WAIC Score:       ", round(waic_val, digits=2))

    println("\n--- MCMC Diagnostics ---")
    println("Mean R-hat:       ", round(mean_rhat, digits=4))
    println("Minimum ESS:      ", round(min_ess, digits=2))
    println("ESS per second:    ", ess_rate)

    # Final Registry Object Assembly
    out = (
        metrics = (
            rmse = rmse_val, 
            r_pearson = r_pearson, 
            waic = waic_val, 
            rhat = mean_rhat, 
            ess = min_ess, 
            ess_rate = ess_rate,
            sampling_time = sampling_time
        ),
        pstats = res,
        plots = plots,
        y_obs = y_obs,
        model_family = model_family,
        arch = arch_type
    )

    return out
end


function model_results_plots(res)
    """
    Displays all plots generated by `model_results_comprehensive` and stored
    in the results object.
    """
    if !hasproperty(res, :plots) || isempty(res.plots)
        println("No plots found in the results object.")
        return
    end

    println("Displaying generated plots...")
    for (plot_name, plot_obj) in pairs(res.plots)
        if plot_obj isa Dict # Handle nested plot dictionaries like for smooth_effects
            for (sub_name, sub_plot) in plot_obj
                println("--- Plot: $plot_name -> $sub_name ---")
                display(sub_plot)
            end
        else
            println("--- Plot: $plot_name ---")
            display(plot_obj)
        end
    end
    println("--- End of plots ---")
end