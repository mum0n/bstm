#=
File: visualization_engine.jl
Version: 1.2.0
Timestamp: 2026-07-01 00:45:00

Description:
This file contains the primary visualization function for the `bstm` framework.
It is responsible for generating a comprehensive dashboard of model results,
including posterior predictive checks and plots for all latent manifold effects.

Changes in this version:
- Created a new dedicated file for visualization logic.
- Corrected keys for spatial effects to `:spatial_denoised` and `:spatial_noisy`.
- Updated `model_results_plots` to plot individual smooth effects against their
  corresponding covariate values, instead of an accumulated sum against the observation index.
  This requires passing the model configuration object `M` to the function.
=#

function model_results_plots(res, M; ts=1; outcome=1, centroids=nothing, polygons=nothing, y_obs=nothing)
    # v1.1.0 (2026-06-30)
    # v1.2.0 (2026-07-01)
    # Purpose: Creates a dashboard of plots for visualizing model results.
    # Change: Plots individual smooths against covariates. Requires model configuration `M`.
    println("--- Rendering Dashboard v19.39.0 [Outcome: ", outcome, "] ---")

    pstats = res.pstats
    obs_src = isnothing(y_obs) ? res.y_obs : y_obs
    plots_list = []

    # Panel 1: Posterior Predictive Check (PPC)
    is_mv = (obs_src isa AbstractMatrix && size(obs_src, 2) > 1)
    y_p = is_mv ? pstats.predictions_denoised.mean[:, outcome] : vec(pstats.predictions_denoised.mean)
    y_o = is_mv ? obs_src[:, outcome] : vec(obs_src)

    if length(y_p) == length(y_o)
        p_ppc = Plots.scatter(vec(y_p), vec(y_o), title="PPC Audit", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
        clean_p = filter(x -> !isnan(x), y_p)
        clean_o = filter(x -> !isnan(x), y_o)
        if !isempty(clean_p) && !isempty(clean_o)
            min_val = min(minimum(clean_p), minimum(clean_o))
            max_val = max(maximum(clean_p), maximum(clean_o))
            Plots.plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5)
        end
        push!(plots_list, p_ppc)
    end

    # Helper function for spatial plots
    function create_spatial_plot(field_data, title_str)
        if !isnothing(field_data) && hasproperty(field_data, :mean)
            s_mean = vec(collect(field_data.mean))
            if !all(iszero, s_mean)
                p_map = Plots.plot(aspect_ratio=:equal, title=title_str, frame=:box, colorbar=true)
                if !isnothing(polygons) && length(polygons) >= length(s_mean)
                    for i in 1:length(s_mean)
                        px = [pt[1] for pt in polygons[i]]
                        py = [pt[2] for pt in polygons[i]]
                        if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                            push!(px, px[1]); push!(py, py[1])
                        end
                        Plots.plot!(p_map, px, py, seriestype=:shape, fill_z=s_mean[i], c=:viridis, linecolor=:black, lw=0.2, label=nothing)
                    end
                elseif !isnothing(centroids)
                    cx = [c[1] for c in centroids]
                    cy = [c[2] for c in centroids]
                    Plots.scatter!(p_map, cx, cy, marker_z=vec(s_mean), markersize=4, c=:viridis, label=nothing)
                end
                return p_map
            end
        end
        return nothing
    end

    # Spatial Plots
    if !isnothing(polygons) || !isnothing(centroids)
        # Denoised (structured) Spatial Effect
        s_field_denoised = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? get(pstats, :spatial_denoised, [nothing])[outcome] : get(pstats, :spatial_denoised, nothing)
        p_spatial_denoised = create_spatial_plot(s_field_denoised, "Spatial Denoised Effect")
        !isnothing(p_spatial_denoised) && push!(plots_list, p_spatial_denoised)

        # Total (noisy) Spatial Effect
        s_field_noisy = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? get(pstats, :spatial_noisy, [nothing])[outcome] : get(pstats, :spatial_noisy, nothing)
        p_spatial_noisy = create_spatial_plot(s_field_noisy, "Total Spatial Effect")
        !isnothing(p_spatial_noisy) && push!(plots_list, p_spatial_noisy)

        # Spacetime Interaction
        st_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? get(pstats, :spacetime_interaction, [nothing])[outcome] : get(pstats, :spacetime_interaction, nothing)
        p_st_interaction = create_spatial_plot(st_field, "Spacetime Interaction (t=$ts)")
        !isnothing(p_st_interaction) && push!(plots_list, p_st_interaction)
    end

    # Temporal Plots
    if hasproperty(pstats, :temporal)
        raw_t = pstats.temporal
        t_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? raw_t[outcome] : raw_t
        if !isnothing(t_field) && hasproperty(t_field, :mean) && !all(iszero, t_field.mean)
            tm, tl, tu = vec(t_field.mean), vec(t_field.lower), vec(t_field.upper)
            p_trend = Plots.plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index")
            push!(plots_list, p_trend)
        end
    end

    # Seasonal Plot
    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal) && hasproperty(pstats.seasonal, :mean) && !all(iszero, pstats.seasonal.mean)
        um, ul, uu = vec(pstats.seasonal.mean), vec(pstats.seasonal.lower), vec(pstats.seasonal.upper)
        p_seas = Plots.plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Cycle", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Season Bin")
        push!(plots_list, p_seas)
    end

    # Mixed Effects Plot
    if hasproperty(pstats, :mixed_effects) && !isnothing(pstats.mixed_effects) && !isempty(pstats.mixed_effects)
        for (grp_sym, m_summ) in pairs(pstats.mixed_effects)
            mm, ml, mu = vec(m_summ.mean), vec(m_summ.lower), vec(m_summ.upper)
            if !all(iszero, mm)
                push!(plots_list, Plots.bar(mm, yerror=(mm .- ml, mu .- mm), title=string("Mixed: ", grp_sym), color=:purple, legend=false, xlabel="Category ID"))
            end
        end
    end

    # Nested Manifold Contributions Plot
    if hasproperty(pstats, :nested_contributions) && !isnothing(pstats.nested_contributions)
        n_field = pstats.nested_contributions
        if hasproperty(n_field, :mean) && !all(iszero, n_field.mean) && Statistics.std(n_field.mean) > 1e-9
            nm, nl, nu = vec(n_field.mean), vec(n_field.lower), vec(n_field.upper)
            push!(plots_list, Plots.plot(nm, ribbon=(nm .- nl, nu .- nm), title="Hierarchical Supervisors", lw=1.5, fillalpha=0.2, color=:brown, legend=false, xlabel="Observation Order"))
        end
    end

    # Smoothed Covariate Effects
    # Individual Smoothed Covariate Effects
    if hasproperty(pstats, :smooth_effects) && pstats.smooth_effects isa Dict
        for (var_sym, smooth_summary) in pstats.smooth_effects
            if hasproperty(M.data, var_sym)
                covariate_data = M.data[!, var_sym]
                
                # Sort by covariate value for a clean plot
                p_order = sortperm(covariate_data)
                
                sm = vec(smooth_summary.mean)
                sl = vec(smooth_summary.lower)
                su = vec(smooth_summary.upper)

                # Plot effect vs covariate value
                p_smooth = Plots.plot(covariate_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]),
                                      title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Effect on eta",
                                      legend=false, color=:darkorange, fillalpha=0.2)
                push!(plots_list, p_smooth)
            end
        end
    end



    # Fixed Effects
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects) && hasproperty(pstats.fixed_effects, :mean)
        fm, fl, fu = vec(pstats.fixed_effects.mean), vec(pstats.fixed_effects.lower), vec(pstats.fixed_effects.upper)
        n_coeffs = length(fm)
        if n_coeffs > 0
            p_forest = Plots.scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), title="Fixed Effects", xlabel="Coefficient", ylabel="Index", markersize=4, color=:black, legend=false, yticks=(1:n_coeffs, ["β$i" for i in 1:n_coeffs]))
            Plots.vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
            push!(plots_list, p_forest)
        end
    end

    # Final plot assembly
    if !isempty(plots_list)
        n_plots = length(plots_list)
        cols = min(n_plots, 2)
        rows = Int(ceil(n_plots / cols))
        final_plt = Plots.plot(plots_list..., layout=(rows, cols), size=(1200, 350 * rows), margin=5Plots.mm)
        return final_plt
    end

    @warn "BSTM Visualization: No active manifolds discovered for outcome $outcome."
    return nothing
end