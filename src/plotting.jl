# Lightweight plotting utilities for bstm
#
# Drop this file into src/ and `include("src/plotting.jl")` (or include from bstm.jl).
# Design goals:
# - Do not overwrite Base/Plots function names.
# - All helpers return a Plots.Plot object (no internal display()).
# - Robust input handling for polygons, centroids, and timeseries.
# - Small, well-documented API: choropleth, timeseries_ci, spatial_graph_plot, render_paths!, map_point_occupancy, save_plot.
#
# Dependencies: Plots, ColorSchemes, Statistics (standard packages already used in the project)
#
# Example:
#   using .bstm  # or include("src/plotting.jl")
#   p = choropleth(polys, vals, title="Risk")
#   save_plot(p, "risk.png")
#
# Notes:
# - Functions accept simple geometry formats used in the repo: polygons = Vector{Vector{<:Tuple{Real,Real}}}
#   centroids = Vector{Tuple{Real,Real}}. If you use Geo types, convert to the expected simple arrays first.

 """
    bstm_plots(model_obj::DynamicPPL.Model, chain, res, M; au=nothing, data=nothing, outcome=1)

Generates a standard set of diagnostic and summary plots from a fitted `bstm` model.

# Version
v1.2.0 (2026-08-20)

# Rationale
This function is the primary visualization engine for the `bstm` framework. It takes
the summarized results from the reconstruction engine and produces a standardized
set of plots for model diagnostics and interpretation. It is designed to be robust
and flexible, correctly handling different model architectures (univariate,
multivariate) and component types (spatial, temporal, smooth, mixed effects).
This version now returns both the plot objects and the underlying data used to
create them.

# Workflow
1.  **Posterior Predictive Check (PPC)**: Creates a scatter plot of observed vs.
    predicted values to assess overall model fit.
2.  **Fixed Effects**: Bar plots of coefficients with credible intervals.
3.  **Conditional Effects**: Plots the expected response for varying values of one or two predictors.
4.  **Component-wise Plotting**: Iterates through all components defined in the model
    configuration (`M.components`).
5.  **Structure-based Dispatch**: For each component, it uses the `structure`
    (e.g., `:spatial`, `:temporal`, `:smooth`) to dispatch to the appropriate
    plotting logic.
    - **Spatial**: Generates choropleth maps (if polygons are provided) or scatter
      plots of the spatial random effects. For `BYM2` models, it creates separate
      plots for the structured and unstructured components.
    - **Temporal/Seasonal**: Creates line plots of the temporal or seasonal trends
      with credible interval ribbons.
    - **Smooth**: Creates line plots showing the non-linear effect of a covariate,
      with credible interval ribbons. For 2D smooths, generates surface/contour plots.
    - **Spatially Varying Coefficients (SVC)**: Generates choropleth/heatmap plots of the estimated coefficient surface.
6.  **Hierarchical Effects**: Visualizes the distribution of group-level parameters.

# Arguments
- `model_obj::DynamicPPL.Model`: The fitted Turing model object.
- `chain`: The MCMC chain object from the fitted model.
- `res`: The results `NamedTuple` from `_reconstruct`, containing summarized effects.
- `M`: The main model configuration `NamedTuple`.
- `au`: An optional object containing areal unit information (`polygons`, `centroids`).
- `data`: The optional input `DataFrame`, used to get coordinate/variable data for axes.
- `outcome`: `Int`, the index of the outcome to plot in a multivariate model.

# Returns
- A `NamedTuple` with two fields:
  - `plots`: A `NamedTuple` where each key corresponds to a plot type and the value is a `Plots.Plot` object or a dictionary of plots.
  - `plots_data`: A `NamedTuple` containing the data used to generate each plot.
"""
function bstm_plots(model_obj::DynamicPPL.Model, chain, res, M; au=nothing, data=nothing, outcome=1)
    plots = Dict{Symbol, Any}()
    plots_data = Dict{Symbol, Any}()
    effects = res.pstats.effects
    is_mv = res.pstats.arch isa MultivariateArchitecture
    
    y_obs = get(M, :y_obs, nothing)

    polygons = if !isnothing(au) && (au isa NamedTuple || au isa Dict)
        get(au, :polygons, nothing)
    else
        nothing
    end
    
    centroids = if !isnothing(au) && (au isa NamedTuple || au isa Dict)
        get(au, :centroids, nothing)
    else
        nothing
    end

    # --- 1. Posterior Predictive Check ---
    if hasproperty(res.pstats, :predictions_denoised)
        if isnothing(y_obs)
            @info "Skipping PPC plot: Observation data not found."
        else
            pred_summary = is_mv ? res.pstats.predictions_denoised[outcome] : res.pstats.predictions_denoised 
            if !isnothing(pred_summary) && hasproperty(pred_summary, :mean)
                y_p, y_o = vec(pred_summary.mean), is_mv ? vec(y_obs[:, outcome]) : vec(y_obs)
                if length(y_p) == length(y_o)
                    p_ppc = scatter(y_p, y_o, title="Posterior Predictive Check", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
                    clean_p, clean_o = filter(!isnan, y_p), filter(!isnan, y_o)
                    if !isempty(clean_p) && !isempty(clean_o)
                        min_val, max_val = min(minimum(clean_p), minimum(clean_o)), max(maximum(clean_p), maximum(clean_o))
                        plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5)
                    end
                    plots[:ppc] = p_ppc
                    plots_data[:ppc] = (predicted = y_p, observed = y_o)
                end
            end
        end
    end

    # --- 2. Helper function for choropleth plots ---
    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if isnothing(field_data) || !hasproperty(field_data, :mean)
            @info "Skipping spatial plot '$title_str': Data missing."
            return nothing
        end 
        if isnothing(polygons) && isnothing(centroids)
            @info "Skipping spatial plot '$title_str': No geometry provided."
            return nothing
        end
        s_mean = vec(collect(field_data.mean))
        if all(iszero, s_mean)
            @info "Skipping spatial plot '$title_str': Mean effect is zero."
            return nothing
        end
        if !isnothing(polygons) && length(polygons) >= length(s_mean)
            return plot_choropleth(s_mean, polygons; title=title_str)
        elseif !isnothing(centroids)
            return scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal)
        end
        return nothing
    end

    # --- 3. Fixed Effects Plots ---
    if hasproperty(effects, :fixed) && !isnothing(effects.fixed)
        fe_summary = is_mv ? effects.fixed[outcome] : effects.fixed
        if hasproperty(fe_summary, :mean) && !all(iszero, fe_summary.mean) 
            fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
            if !isempty(fm)
                coef_names = haskey(M, :Xfixed_names) ? string.(M.Xfixed_names) : ["Coef_$i" for i in 1:length(fm)]
                p_forest = scatter(fm, 1:length(fm), xerror=(fm .- fl, fu .- fm), yticks=(1:length(fm), coef_names), title="Fixed Effects Coefficients", xlabel="Estimate", markersize=4, color=:black, legend=false)
                vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                plots[:fixed_effects] = p_forest
                plots_data[:fixed_effects] = (names=coef_names, mean=fm, lower=fl, upper=fu)
            end
        end
    end

    # --- 4. Conditional Effects Plots ---
    conditional_plots = Dict{Symbol, Any}()
    conditional_plots_data = Dict{Symbol, Any}()
    all_covariates = String[]
    if haskey(M, :Xfixed_names); append!(all_covariates, string.(M.Xfixed_names)); end
    for spec in M.components
        if spec.structure == :smooth
            vars = get(spec.params, :positional_args, [])
            append!(all_covariates, string.(vars))
        end
    end
    all_covariates = unique(Symbol.(all_covariates))

    for cov_sym in all_covariates
        if !hasproperty(M.data, cov_sym); continue; end

        if eltype(M.data[!, cov_sym]) <: Number # Continuous covariate
            cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, cov_sym)
            if !isnothing(cond_preds_res)
                cond_preds, cov_range = cond_preds_res
                cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                
                cm, cl, cu = vec(cond_summary.mean), vec(cond_summary.lower), vec(cond_summary.upper)
                p_cond = plot(cov_range, cm, ribbon=(cm .- cl, cu .- cm), title="Conditional Effect: $(cov_sym)", xlabel=string(cov_sym), ylabel="Expected Response", lw=2, fillalpha=0.2, color=:blue, legend=false)
                plot_key = Symbol("conditional_$(cov_sym)")
                conditional_plots[plot_key] = p_cond
                conditional_plots_data[plot_key] = (covariate_values=cov_range, mean=cm, lower=cl, upper=cu)
            end
        else # Categorical covariate
            cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, cov_sym)
            if !isnothing(cond_preds_res)
                cond_preds, cov_levels = cond_preds_res
                cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                
                cm, cl, cu = vec(cond_summary.mean), vec(cond_summary.lower), vec(cond_summary.upper)
                p_cond = bar(string.(cov_levels), cm, yerror=(cm .- cl, cu .- cm), title="Conditional Effect: $(cov_sym)", xlabel=string(cov_sym), ylabel="Expected Response", color=:blue, legend=false)
                plot_key = Symbol("conditional_$(cov_sym)")
                conditional_plots[plot_key] = p_cond
                conditional_plots_data[plot_key] = (covariate_levels=string.(cov_levels), mean=cm, lower=cl, upper=cu)
            end
        end
    end
    if !isempty(conditional_plots)
        plots[:conditional_effects] = conditional_plots
        plots_data[:conditional_effects] = conditional_plots_data
    end

    # --- 5. Component-wise Plotting ---
    smooth_effects_plots = Dict{Symbol, Any}()
    smooth_effects_plots_data = Dict{Symbol, Any}()
    for spec in M.components
        key = spec.key
        
        if spec.component_obj isa Mixed; continue; end

        if !haskey(effects, key)
            @info "Skipping plot for component '$key': No effect summary found in results."
            continue
        end
        
        component_effects = effects[key]
        
        main_effect_summary = if hasproperty(component_effects, :noisy)
            is_mv ? component_effects.noisy[outcome] : component_effects.noisy
        elseif hasproperty(component_effects, :structured)
            is_mv ? component_effects.structured[outcome] : component_effects.structured
        else
            continue
        end

        if isnothing(main_effect_summary) || !hasproperty(main_effect_summary, :mean) || all(iszero, main_effect_summary.mean)
            @info "Skipping plot for component '$key': Main effect is zero or data is missing."
            continue
        end

        if spec.structure == :spatial
            plot_key = Symbol("spatial_$(key)")
            p = _create_choropleth_plot(main_effect_summary, "Spatial Effect: $key", polygons, centroids)
            if !isnothing(p)
                plots[plot_key] = p
                plots_data[plot_key] = (values=vec(main_effect_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
            end

            if hasproperty(component_effects, :structured)
                struct_summary = is_mv ? component_effects.structured[outcome] : component_effects.structured
                p_struct = _create_choropleth_plot(struct_summary, "Structured Effect: $key", polygons, centroids)
                if !isnothing(p_struct)
                    plot_key_struct = Symbol("structured_$(key)")
                    plots[plot_key_struct] = p_struct
                    plots_data[plot_key_struct] = (values=vec(struct_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                end
            end
            if hasproperty(component_effects, :unstructured)
                unstruct_summary = is_mv ? component_effects.unstructured[outcome] : component_effects.unstructured
                p_unstruct = _create_choropleth_plot(unstruct_summary, "Unstructured Effect: $key", polygons, centroids)
                if !isnothing(p_unstruct)
                    plot_key_unstruct = Symbol("unstructured_$(key)")
                    plots[plot_key_unstruct] = p_unstruct
                    plots_data[plot_key_unstruct] = (values=vec(unstruct_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                end
            end

        elseif spec.structure == :temporal
            plot_key = Symbol("temporal_$(key)")
            if !isnothing(data) && haskey(M, :t_idx_var) && hasproperty(data, M.t_idx_var)
                time_var = M.t_idx_var
                time_coords = data[!, time_var]
                p_order = sortperm(time_coords)
                tm, tl, tu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                plots[plot_key] = plot(time_coords[p_order], tm[p_order], ribbon=(tm[p_order] .- tl[p_order], tu[p_order] .- tm[p_order]), title="Temporal Trend: $key", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel=string(time_var))
                plots_data[plot_key] = (time=time_coords[p_order], mean=tm[p_order], lower=tl[p_order], upper=tu[p_order])
            else
                tm, tl, tu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                plots[plot_key] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend: $key", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index")
                plots_data[plot_key] = (time=1:length(tm), mean=tm, lower=tl, upper=tu)
            end

        elseif spec.structure == :seasonal
            plot_key = Symbol("seasonal_$(key)")
            um, ul, uu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
            plots[plot_key] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Component: $key", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Period")
            plots_data[plot_key] = (period=1:length(um), mean=um, lower=ul, upper=uu)

        elseif spec.structure == :smooth
            if isnothing(data); @info "Skipping smooth effect plot for '$key': `data` not provided."; continue; end
            
            vars = get(spec.params, :positional_args, [])
            if length(vars) == 1 # 1D smooth
                var_sym = Symbol(vars[1])
                if hasproperty(data, var_sym)
                    cov_data = data[!, var_sym]
                    p_order = sortperm(cov_data)
                    sm, sl, su = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                    
                    smooth_effects_plots[var_sym] = plot(cov_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]), title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Latent Effect", legend=false, color=:darkorange, fillalpha=0.2)
                    smooth_effects_plots_data[var_sym] = (covariate_values=cov_data[p_order], mean=sm[p_order], lower=sl[p_order], upper=su[p_order])
                end
            elseif length(vars) == 2 # 2D smooth (interaction)
                var1_sym, var2_sym = Symbol(vars[1]), Symbol(vars[2])
                if hasproperty(data, var1_sym) && hasproperty(data, var2_sym)
                    cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, var1_sym, second_cov=var2_sym)
                    if !isnothing(cond_preds_res)
                        cond_preds, range1, range2 = cond_preds_res
                        cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                        
                        grid_mean = reshape(vec(cond_summary.mean), length(range1), length(range2))
                        
                        plot_key = Symbol("$(var1_sym)_$(var2_sym)")
                        smooth_effects_plots[plot_key] = heatmap(range1, range2, grid_mean', title="2D Smooth Effect: $(var1_sym) & $(var2_sym)", xlabel=string(var1_sym), ylabel=string(var2_sym), c=:viridis, legend=false)
                        smooth_effects_plots_data[plot_key] = (x=range1, y=range2, z=grid_mean)
                    end
                end
            end
        end
    end
    if !isempty(smooth_effects_plots)
        plots[:smooth_effects] = smooth_effects_plots
        plots_data[:smooth_effects] = smooth_effects_plots_data
    end

    # --- 6. Spatiotemporal Interaction Effects ---
    if hasproperty(effects, :st_interaction) && !isnothing(effects.st_interaction)
        st_summary = is_mv ? effects.st_interaction[outcome] : effects.st_interaction
        if hasproperty(st_summary, :mean) && !all(iszero, st_summary.mean)
            if haskey(M, :s_N) && haskey(M, :t_N)
                st_mean_grid = reshape(vec(st_summary.mean), M.s_N, M.t_N)
                p_st_heatmap = heatmap(1:M.t_N, 1:M.s_N, st_mean_grid, title="Spatiotemporal Interaction", xlabel="Time Index", ylabel="Spatial Unit Index", c=:viridis, legend=false)
                plots[:st_interaction_heatmap] = p_st_heatmap
                plots_data[:st_interaction_heatmap] = (time_idx=1:M.t_N, space_idx=1:M.s_N, mean_effect=st_mean_grid)
            else
                @warn "Skipping spatiotemporal interaction plot: M.s_N or M.t_N not found."
            end
        end
    end

    # --- 7. Spatially Varying Coefficients (SVC) Plots ---
    for spec in M.components
        if spec.structure == :svc
            key = spec.key
            if !haskey(effects, key) || (isnothing(polygons) && isnothing(centroids)); continue; end
            
            svc_effect_summary = is_mv ? effects[key].structured[outcome] : effects[key].structured
            if !isnothing(svc_effect_summary) && hasproperty(svc_effect_summary, :mean)
                plot_key = Symbol("svc_$(key)")
                p_svc = _create_choropleth_plot(svc_effect_summary, "SVC Effect: $(key)", polygons, centroids)
                if !isnothing(p_svc)
                    plots[plot_key] = p_svc
                    plots_data[plot_key] = (values=vec(svc_effect_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                end
            end
        end
    end

    # --- 8. Hierarchical Effects (Mixed Effects) Plots ---
    if hasproperty(effects, :mixed_effects) && !isnothing(effects.mixed_effects)
        mixed_plots = Dict{Symbol, Any}()
        mixed_plots_data = Dict{Symbol, Any}()
        for (key, effect_summary) in pairs(effects.mixed_effects)
            group_var = Symbol(effect_summary.group_var)
            group_levels = hasproperty(M.data, group_var) ? unique(M.data[!, group_var]) : nothing

            summaries_to_plot = is_mv ? effect_summary.summaries[outcome] : effect_summary.summaries

            for (term_name, summary) in pairs(summaries_to_plot)
                if hasproperty(summary, :mean) && !all(iszero, summary.mean)
                    means = vec(summary.mean)
                    lowers = vec(summary.lower)
                    uppers = vec(summary.upper)
                    n_levels = length(means) 
                    
                    y_ticks_labels = isnothing(group_levels) || length(group_levels) != n_levels ? ["Level $i" for i in 1:n_levels] : string.(group_levels)
                    
                    p_title = "Mixed Effect: $(term_name) | $(group_var)"
                    
                    p_forest = scatter(means, 1:n_levels, xerror=(means .- lowers, uppers .- means), yticks=(1:n_levels, y_ticks_labels), title=p_title, xlabel="Effect Size", markersize=4, color=:black, legend=false, yflip=true)
                    vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                    
                    plot_key = Symbol("$(key)_$(term_name)")
                    mixed_plots[plot_key] = p_forest
                    mixed_plots_data[plot_key] = (group_levels=y_ticks_labels, mean=means, lower=lowers, upper=uppers)
                end
            end
        end
        if !isempty(mixed_plots)
            plots[:mixed_effects] = mixed_plots
            plots_data[:mixed_effects] = mixed_plots_data
        end
    end

    return (plots=NamedTuple(plots), plots_data=NamedTuple(plots_data))
end




"""
    model_results_plots(res)

A convenience function to display all plots generated by the `model_results_comprehensive`
function.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function provides a simple and standardized way to visualize all the output
plots from a `bstm` model run. It is designed to handle the nested structure of the
`plots` object returned by `model_results_comprehensive`, which may contain both
individual plot objects and dictionaries of plots (for example, for multiple smooth or
mixed effects).

# Arguments
- `res`: The main results `NamedTuple` returned by `model_results_comprehensive`.

# Returns
- `nothing`. The function prints the plots to the current display.
"""
function model_results_plots(res)
    if !hasproperty(res, :plots) || isempty(res.plots)
        println("No plots found in the results object.") 
        return
    end

    println("--- Displaying Generated Plots ---")
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
    println("--- End of Plots ---")
end


"""
    plot_choropleth(values::AbstractVector, polygons::Vector; title="Spatial Distribution", cmap=:viridis)

A utility function to generate a choropleth map from a set of values and their
corresponding spatial polygons.

# Version
v1.0.1 (2026-08-13)

# Rationale
This function provides a standardized and simple way to visualize spatial data,
which is a core requirement for interpreting the outputs of spatiotemporal models.
It is used by the main `bstm_plots` function to create maps of spatial random
effects and predictions. The implementation is robust, handling common issues like
non-closed polygon shapes and invalid coordinate data.

# Arguments
- `values::AbstractVector`: A vector of numeric values, where each value corresponds
  to a polygon.
- `polygons::Vector`: A vector of polygons. Each element of the vector should be a
  collection of points (e.g., a `Vector` of 2-element `Tuple`s or `Vector`s) that
  define the vertices of a polygon.
- `title::String`: The title for the plot.
- `cmap`: The colormap to use for shading the polygons. Can be a `Symbol` (e.g.,
  `:viridis`) or a `ColorGradient`.

# Returns
- A `Plots.Plot` object representing the choropleth map.
"""
function plot_choropleth(values::AbstractVector, polygons::Vector; title="Spatial Distribution", cmap=:viridis)
    plt = plot(aspect_ratio=:equal, title=title, legend=false, grid=false, showaxis=false, xticks=false, yticks=false)

    # Determine the color range for normalization, handled automatically by Plots.jl
    # but useful to have if manual normalization were needed.
    min_val, max_val = extrema(values)
    
    for i in 1:min(length(polygons), length(values))
        poly_coords = polygons[i]
        
        # A valid polygon requires at least 3 vertices.
        if length(poly_coords) > 2 
            # Extract x and y coordinates, filtering out any NaN values.
            px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
            py = [pt[2] for pt in poly_coords if !isnan(pt[2])]
            
            # Proceed only if there are valid coordinates.
            if !isempty(px)
                # Ensure the polygon is closed for plotting.
                if (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end
                
                plot!(plt, px, py, seriestype=:shape, fill_z=values[i], c=cmap, linecolor=:black, lw=0.5, fillalpha=0.8, label=nothing) 
            end
        end
    end
    return plt
end



function plot_spatial_graph(; au=nothing, pts=nothing, plot_title="Spatial Partitioning")
    """
    BSTM Partitioning Utility v1.0.0
    Timestamp: 2026-06-26 10:01:50
    Synopsis: Visualizes the results of a spatial partitioning. It plots the generated polygons,
              the adjacency graph, the centroids, the overall hull, and optionally the raw data points.
    Inputs:
        - au: The `areal_units` object returned by `assign_spatial_units`.
        - pts: Optional. A vector of (x, y) tuples representing the raw data points to overlay.
        - plot_title: The title for the plot.
    Outputs:
        - A `Plots.Plot` object.
    """
    # 2. Base Plot Initialization
    plt = Plots.plot(aspect_ratio=:equal, legend=false)
    Plots.title!(plt, plot_title)

    # 3. Polygon Geometry Rendering
        for poly_coords in au[:polygons]
            if length(poly_coords) > 2
                px = [p[1] for p in poly_coords if !isnan(p[1])]
                py = [p[2] for p in poly_coords if !isnan(p[2])]
                if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end
                Plots.plot!(plt, px, py, seriestype=:shape, fillalpha=0.1, linecolor=:black, lw=0.5)
            end
        end 

    # 4. Adjacency Graph Edge Rendering 
        for edge in Graphs.edges(au[:graph])
            u, v = Graphs.src(edge), Graphs.dst(edge)
            p1, p2 = au[:centroids][u], au[:centroids][v]
            Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
        end 

    # 5. Scatter Plotting: Data Points and Polygon Centroids
    if !isnothing(pts)
        Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
            markersize=1, color=:gray, alpha=0.3, label="Points")
    end
 
        Plots.scatter!(plt, [c[1] for c in au[:centroids]], [c[2] for c in au[:centroids]],
            markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    # 6. Boundary Constraints
        bx = [p[1] for p in au[:hull_coords] if !isnan(p[1])]
        by = [p[2] for p in au[:hull_coords] if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)

    return plt
end


# ---------------------------------------------------------------------
# Utility helpers (internal)
# ---------------------------------------------------------------------
# Convert polygon representation into vectors of x and y coordinates.
# Accepts: Vector{<:Tuple{Real,Real}} or Vector{<:AbstractVector} where elements are (x,y).
function _poly_xy(poly)
    xs = Float64[]
    ys = Float64[]
    for p in poly
        push!(xs, float(p[1]))
        push!(ys, float(p[2]))
    end
    return xs, ys
end

# Safe extraction of centroids list => two vectors
function _centroids_xy(centroids)
    xs = Float64[]
    ys = Float64[]
    for c in centroids
        push!(xs, float(c[1]))
        push!(ys, float(c[2]))
    end
    return xs, ys
end

# Get a ColorScheme object for use with get(cs, t)
function _get_cscheme(cmap)
    try
        return get(ColorSchemes, cmap, ColorSchemes.viridis)
    catch
        return ColorSchemes.viridis
    end
end

# Map a scalar value to a color from a ColorScheme (linear mapping)
function _map_color_from_range(cscheme, v, vmin, vmax)
    if isnan(v)
        return RGB(0.9, 0.9, 0.9)  # NaN -> light gray
    end
    t = (v - vmin) / (vmax - vmin + eps())  # eps avoids div-by-zero
    t = clamp(t, 0.0, 1.0)
    return get(cscheme, t)
end

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

"""
    create_theme(; fontsize=10, titlefontsize=12, font="DejaVu Sans", size=(900,600))

Return a Dict of common plotting defaults (use with `plot(...; create_theme()...)`).
Does not apply global state; caller can pass the kwargs into plotting functions.
"""
function create_theme(; fontsize=10, titlefontsize=12, font="DejaVu Sans", size=(900,600))
    return (; legendfontsize=fontsize, guidefontsize=fontsize,
             tickfontsize=max(8, fontsize-1), titlefontsize=titlefontsize,
             fontfamily=font, size=size)
end


"""
    choropleth(polygons::Vector, values::AbstractVector;
               cmap=:viridis, vmin=nothing, vmax=nothing, center_zero=false,
               show_colorbar=true, title="", border_color=:black, border_alpha=0.6,
               lw=0.4, closed=true, theme_kwargs=NamedTuple(), colorbar_label=nothing)

Create a choropleth plot from `polygons` (Vector of polygon coords) and `values`.
polygons: e.g. Vector{Vector{Tuple{x,y}}} or Vector{Vector{<:AbstractVector}}.
values: numeric vector aligned to polygons.

Returns: a Plots.Plot (does not call display()).
"""
function choropleth(polygons::Vector, values::AbstractVector;
                    cmap=:viridis, vmin=nothing, vmax=nothing, center_zero::Bool=false,
                    show_colorbar::Bool=true, title::String="", border_color=:black,
                    border_alpha::Real=0.6, lw::Real=0.4, closed::Bool=true,
                    theme_kwargs=NamedTuple(), colorbar_label::Union{Nothing,String}=nothing)

    n = length(polygons)
    @assert length(values) == n "length(values) must equal length(polygons)"

    # canonical numeric array
    vals = collect(Float64, values)
    mask = .!isnan.(vals)

    # If all NaN: return polygon outlines
    if all(!mask)
        p = plot(aspect_ratio=:equal, title=title, legend=false; theme_kwargs...)
        for poly in polygons
            if length(poly) < 3; continue; end
            xs, ys = _poly_xy(poly)
            if closed && (xs[1], ys[1]) != (xs[end], ys[end])
                push!(xs, xs[1]); push!(ys, ys[1])
            end
            plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=border_color, lw=lw, alpha=border_alpha, label=nothing)
        end
        return p
    end

    # Robust vmin/vmax: use 2nd and 98th percentiles to reduce outlier influence unless user supplies them
    cs = _get_cscheme(cmap)
    vvals = vals[mask]
    if vmin === nothing; vmin = quantile(vvals, 0.02); end
    if vmax === nothing; vmax = quantile(vvals, 0.98); end

    if center_zero
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
        cs = _get_cscheme(:RdBu)
    end

    p = plot(aspect_ratio=:equal, title=title, legend=false; theme_kwargs...)

    # fill polygons
    for (i, poly) in enumerate(polygons)
        if length(poly) < 3; continue; end
        xs, ys = _poly_xy(poly)
        if closed && (xs[1], ys[1]) != (xs[end], ys[end])
            push!(xs, xs[1]); push!(ys, ys[1])
        end
        fillcolor = mask[i] ? _map_color_from_range(cs, vals[i], vmin, vmax) : RGB(0.9,0.9,0.9)
        plot!(p, xs, ys, seriestype=:shape, fillcolor=fillcolor, linecolor=border_color, lw=lw, alpha=0.95, label=nothing)
    end

    # colorbar: Plots.jl doesn't expose an easy standalone colorbar for shapes,
    # so we create an invisible scatter to force a colorbar.
    if show_colorbar
        cb_vals = range(vmin, stop=vmax, length=64)
        scatter!(p, zeros(length(cb_vals)), ones(length(cb_vals)), zcolor=cb_vals,
                 markersize=0.1, c=cs, colorbar=true, label=nothing)
        if colorbar_label !== nothing
            plot!(p, colorbar_title=colorbar_label)
        end
    end

    return p
end


"""
    timeseries_ci(x, mean, lower, upper; color=:royalblue, title="", xlabel="", theme_kwargs=NamedTuple())

Plot a timeseries with a shaded credible/confidence interval (ribbon). Returns a Plots.Plot.
- x may be missing/empty: if so, uses 1:length(mean).
"""
function timeseries_ci(x, mean, lower, upper; color=:royalblue, title::String="", xlabel::String="", theme_kwargs=NamedTuple())
    meanv = collect(Float64, mean)
    lowv = collect(Float64, lower)
    upv = collect(Float64, upper)

    if isempty(x)
        xvals = collect(1:length(meanv))
    else
        xvals = x
    end

    p = plot(xvals, meanv, ribbon=(meanv .- lowv, upv .- meanv),
             color=color, lw=2, fillalpha=0.2, legend=false,
             title=title, xlabel=xlabel; theme_kwargs...)

    return p
end


"""
    spatial_graph_plot(centroids, g::Graphs.AbstractGraph; polygons=nothing,
                       node_size=3, node_color=:black, edge_color=:red, edge_alpha=0.6,
                       title="", theme_kwargs=NamedTuple())

Plot polygon outlines (optional) and draw adjacency edges between centroids according to Graphs graph `g`.
centroids: Vector of (x,y).
"""
function spatial_graph_plot(centroids, g; polygons=nothing, node_size=3, node_color=:black,
                            edge_color=:red, edge_alpha=0.6, title::String="", theme_kwargs=NamedTuple())

    p = plot(aspect_ratio=:equal, title=title, legend=false; theme_kwargs...)

    # polygons (optional)
    if polygons !== nothing
        for poly in polygons
            if length(poly) < 3; continue; end
            xs, ys = _poly_xy(poly)
            plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:grey, lw=0.4, alpha=0.4, label=nothing)
        end
    end

    # edges
    using Graphs
    for e in edges(g)
        u, v = src(e), dst(e)
        p1 = centroids[u]; p2 = centroids[v]
        plot!(p, [p1[1], p2[1]], [p1[2], p2[2]], color=edge_color, lw=1.2, alpha=edge_alpha, label=nothing)
    end

    # nodes
    xs, ys = _centroids_xy(centroids)
    scatter!(p, xs, ys, markersize=node_size, color=node_color, label=nothing)

    return p
end


"""
    render_paths!(p::Plots.Plot, paths; labels=nothing, color=:black, lw=1.0, markersize=2.0)

Add one-or-more paths to an existing plot `p`. `paths` can be:
- Vector of Vector of indices/centroid tuples: [[(x1,y1), (x2,y2), ...], ...]
- Matrix: each row is a path of indices into `centroids` (use alternative helper if needed)

This helper mutates `p` and returns it.
"""
function render_paths!(p::Plots.Plot, paths; labels=nothing, color=:black, lw=1.0, markersize=2.0)
    # Accept single path
    if isa(paths, AbstractVector) && !isempty(paths) && !isa(paths[1], Number) && isa(paths[1], AbstractVector)
        # Vector of paths (each path is vector of points)
        for (i, path) in enumerate(paths)
            xs = [pt[1] for pt in path]
            ys = [pt[2] for pt in path]
            lab = (labels === nothing) ? nothing : labels[i]
            plot!(p, xs, ys, marker=:circle, markersize=markersize, lw=lw, label=lab, color=color)
        end
        return p
    end

    # Unsupported shape: raise informative error
    error("render_paths! expects paths as Vector of Vector of (x,y) points.")
end


"""
    map_point_occupancy(polygons, centroids, id, step; highlight_color=:red, title="")

Produce a map highlighting the polygon occupied by `id` at `step`. `polygons` and `centroids`
are expected, and `id` may be an integer index into polygons/centroids. Returns a Plot.
"""
function map_point_occupancy(polygons, centroids, id::Integer, step::Integer; highlight_color=:red, title::String="")
    p = plot(aspect_ratio=:equal, legend=false, title=title)

    # draw polygons lightly
    for poly in polygons
        if length(poly) < 3; continue; end
        xs, ys = _poly_xy(poly)
        plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:gray, lw=0.3, alpha=0.6, label=nothing)
    end

    # highlight the polygon (id)
    if 1 <= id <= length(polygons)
        poly = polygons[id]
        xs, ys = _poly_xy(poly)
        plot!(p, xs, ys, seriestype=:shape, fillcolor=highlight_color, linecolor=:black, lw=0.8, alpha=0.5, label=nothing)
    end

    # plot centroids
    xs, ys = _centroids_xy(centroids)
    scatter!(p, xs, ys, markersize=3, color=:black, label=nothing)

    return p
end


"""
    save_plot(p::Plots.Plot, path::AbstractString; fmt=nothing, dpi=150)

Save a plot to disk. If fmt is provided it is used as the file extension, otherwise
the extension of `path` is used.
"""
function save_plot(p::Plots.Plot, path::AbstractString; fmt=nothing, dpi::Integer=150)
    # ensure directory exists
    dir = dirname(path)
    try
        isdir(dir) || mkpath(dir)
    catch
        # ignore if mkpath fails; savefig will error later with informative message
    end

    if fmt !== nothing
        savefig(p, string(path, ".", Symbol(fmt)))
    else
        savefig(p, path)
    end
    return path
end

 