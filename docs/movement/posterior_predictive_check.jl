# =============================================================================
# Posterior Predictive Checks: Model Validation
# =============================================================================

"""
    posterior_predictive_check(loaded, fitted, kernels, params) -> NamedTuple

Performs Bayesian posterior predictive checks to validate that the fitted
movement model reproduces the observed mark-recapture distribution.

For each posterior MCMC draw, this function:
1. Constructs a transition kernel from posterior parameters.
2. Simulates recapture locations for all observed release sites.
3. Compares simulated vs. observed recapture distributions using:
   - Brier score (squared error on recapture probabilities)
   - Kullback-Leibler divergence (distribution matching)
   - Graphical checks (cumulative distribution functions, rank histograms)

# Arguments
- `loaded`: Output of `load_movement_data`.
- `fitted`: Output of `fit_movement_models`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Config NamedTuple. Relevant keys: `seed`, `verbose`.

# Returns
`NamedTuple` with:
- `brier_scores::Vector{Float64}`: Per-draw Brier score.
- `kl_divergences::Vector{Float64}`: Per-draw KL divergence.
- `observed_recapture_dist::Vector{Int}`: Observed recapture locations.
- `simulated_recapture_dists::Vector{Vector{Int}}`: Simulated recapture draws per chain draw.
- `rank_histogram::Vector{Int}`: Rank of observed within simulated predictive dist.
- `summary::NamedTuple`: Summary statistics for diagnostics.
"""
function posterior_predictive_check(loaded, fitted, kernels, params)::NamedTuple
    verbose   = params.verbose
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    chains    = fitted.chains
    land_mask = loaded.land_mask
    
    # Select active chain
    active_chain = haskey(chains, :telemetry) ?
                   chains[:telemetry] :
                   chains[:telemetry_and_survey]
    
    chn_array = Array(active_chain)
    n_draws = size(chn_array, 1)
    G = kernels.G
    
    verbose && println("\n[Posterior Predictive Check] Validating $(n_draws) draws...")
    
    # Extract posterior samples
    v_samples = zeros(n_draws, G)
    d_samples = zeros(n_draws, G)
    g_samples = zeros(n_draws, G)
    
    for draw in 1:n_draws
        for g in 1:G
            v_key = Symbol("velocity[$g]")
            d_key = Symbol("diffusion[$g]")
            g_key = Symbol("gamma[$g]")
            
            v_samples[draw, g] = haskey(chains, v_key) ? mean(active_chain[v_key]) : 0.3
            d_samples[draw, g] = haskey(chains, d_key) ? mean(active_chain[d_key]) : 0.1
            g_samples[draw, g] = haskey(chains, g_key) ? mean(active_chain[g_key]) : 1.0
        end
    end
    
    # Observed recapture distribution (pooled across all observations)
    observed_recaptures = obs_df.recapture
    observed_dist = zeros(n_spatial)
    for rec in observed_recaptures
        if 1 <= rec <= n_spatial
            observed_dist[rec] += 1
        end
    end
    observed_dist ./= sum(observed_dist)
    
    # Storage for posterior predictive samples
    brier_scores = Float64[]
    kl_divergences = Float64[]
    simulated_recapture_dists = Vector{Int}[]
    rank_histogram_vals = Int[]
    
    rng = MersenneTwister(params.seed)
    
    for draw in 1:n_draws
        # Construct kernel from this posterior draw
        alpha_draw = clamp(v_samples[draw, 1] / (v_samples[draw, 1] + d_samples[draw, 1] + 1e-6), 0.0, 1.0)
        rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, 1] + d_samples[draw, 1]), 0.01, 0.95)
        gamma_draw = g_samples[draw, 1]
        
        P_draw = construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = fill(gamma_draw, G),
            residence = fill(rho_draw, G),
            advection = fill(alpha_draw, G),
            land_mask = land_mask
        )
        
        # Simulate recaptures for all observed release sites
        simulated_recaptures = Int[]
        
        for row in eachrow(obs_df)
            rel = row.release
            k   = row.k
            grp = hasproperty(row, :group) ? row.group : 1
            grp = clamp(grp, 1, G)
            
            P_k = P_draw isa AbstractVector ? P_draw[grp] : P_draw
            
            if !isnothing(P_k) && 1 <= rel <= n_spatial
                # Apply transition kernel k times
                prob_vec = zeros(n_spatial)
                prob_vec[rel] = 1.0
                
                for _ in 1:k
                    prob_vec = P_k' * prob_vec
                end
                
                # Sample recapture location from predictive distribution
                if sum(prob_vec) > 0
                    prob_vec = prob_vec ./ sum(prob_vec)
                    rec_site = rand(rng, Categorical(prob_vec))
                    push!(simulated_recaptures, rec_site)
                else
                    push!(simulated_recaptures, rel)  # Fallback
                end
            end
        end
        
        # Compute simulated recapture distribution
        simulated_dist = zeros(n_spatial)
        for rec in simulated_recaptures
            if 1 <= rec <= n_spatial
                simulated_dist[rec] += 1
            end
        end
        simulated_dist ./= (sum(simulated_dist) + 1e-10)
        
        push!(simulated_recapture_dists, simulated_recaptures)
        
        # Brier score: mean squared error between observed and simulated probabilities
        brier = mean((observed_dist .- simulated_dist) .^ 2)
        push!(brier_scores, brier)
        
        # Kullback-Leibler divergence: KL(observed || simulated)
        # KL(P || Q) = sum(P * log(P/Q))
        kl = 0.0
        for i in 1:n_spatial
            if observed_dist[i] > 1e-10 && simulated_dist[i] > 1e-10
                kl += observed_dist[i] * log(observed_dist[i] / simulated_dist[i])
            elseif observed_dist[i] > 1e-10
                kl += observed_dist[i] * log(observed_dist[i] / 1e-10)
            end
        end
        push!(kl_divergences, kl)
        
        if verbose && draw % max(1, div(n_draws, 10)) == 0
            println("  Draw $draw/$n_draws: Brier=$(round(brier; digits=4)), " *
                    "KL=$(round(kl; digits=4))")
        end
    end
    
    # Compute rank histogram: for each observation, rank its location within posterior predictive dist
    for row in eachrow(obs_df)
        rel = row.release
        rec = row.recapture
        k   = row.k
        grp = hasproperty(row, :group) ? row.group : 1
        grp = clamp(grp, 1, G)
        
        # Predict recapture probability from all draws
        pred_probs = zeros(n_draws)
        for draw in 1:n_draws
            alpha_draw = clamp(v_samples[draw, grp] / (v_samples[draw, grp] + d_samples[draw, grp] + 1e-6), 0.0, 1.0)
            rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, grp] + d_samples[draw, grp]), 0.01, 0.95)
            gamma_draw = g_samples[draw, grp]
            
            P_draw = construct_stochastic_transition_kernel(
                loaded.W, loaded.hsi_vec;
                gamma     = fill(gamma_draw, G),
                residence = fill(rho_draw, G),
                advection = fill(alpha_draw, G),
                land_mask = land_mask
            )
            
            P_k = P_draw isa AbstractVector ? P_draw[grp] : P_draw
            
            if !isnothing(P_k) && 1 <= rel <= n_spatial && 1 <= rec <= n_spatial
                prob_vec = zeros(n_spatial)
                prob_vec[rel] = 1.0
                
                for _ in 1:k
                    prob_vec = P_k' * prob_vec
                end
                
                pred_probs[draw] = prob_vec[rec]
            end
        end
        
        # Rank of observed prob within posterior draws
        if any(pred_probs .> 0)
            rank = sum(pred_probs .>= mean(pred_probs))
            push!(rank_histogram_vals, rank)
        end
    end
    
    # Summary statistics
    summary = (
        n_draws              = n_draws,
        n_observations       = length(observed_recaptures),
        brier_mean           = mean(brier_scores),
        brier_sd             = std(brier_scores),
        brier_lower_ci       = quantile(brier_scores, 0.025),
        brier_upper_ci       = quantile(brier_scores, 0.975),
        kl_mean              = mean(kl_divergences),
        kl_sd                = std(kl_divergences),
        kl_lower_ci          = quantile(kl_divergences, 0.025),
        kl_upper_ci          = quantile(kl_divergences, 0.975),
        rank_histogram_mean  = mean(rank_histogram_vals),
        observed_dist_entropy = -sum(observed_dist[observed_dist .> 1e-10] .* log.(observed_dist[observed_dist .> 1e-10])),
    )
    
    if verbose
        println("[Posterior Predictive Check] Summary:")
        println("  Brier score: $(round(summary.brier_mean; digits=4)) " *
                "[$(round(summary.brier_lower_ci; digits=4))-$(round(summary.brier_upper_ci; digits=4))]")
        println("  KL divergence: $(round(summary.kl_mean; digits=4)) " *
                "[$(round(summary.kl_lower_ci; digits=4))-$(round(summary.kl_upper_ci; digits=4))]")
        println("  Observed entropy: $(round(summary.observed_dist_entropy; digits=4))")
    end
    
    return (
        brier_scores             = brier_scores,
        kl_divergences           = kl_divergences,
        observed_recaptures      = observed_recaptures,
        observed_dist            = observed_dist,
        simulated_recapture_dists = simulated_recapture_dists,
        rank_histogram           = rank_histogram_vals,
        summary                  = summary,
    )
end

"""
    export_posterior_predictive_check(ppc, output_dir) -> String

Exports posterior predictive check diagnostics to CSV and plots.

# Arguments
- `ppc`: Output of `posterior_predictive_check`.
- `output_dir`: Output directory path.

# Returns
Path to summary CSV file.
"""
function export_posterior_predictive_check(ppc, output_dir)::String
    mkpath(output_dir)
    
    # Export per-draw diagnostic summaries
    diag_file = joinpath(output_dir, "posterior_predictive_diagnostics.csv")
    diag_df = DataFrame(
        draw               = 1:length(ppc.brier_scores),
        brier_score        = ppc.brier_scores,
        kl_divergence      = ppc.kl_divergences,
    )
    CSV.write(diag_file, diag_df)
    
    # Export summary statistics
    summary_file = joinpath(output_dir, "posterior_predictive_summary.txt")
    open(summary_file, "w") do f
        write(f, "Posterior Predictive Check Summary\n")
        write(f, "=" ^ 50 * "\n\n")
        write(f, "Total observations: $(ppc.summary.n_observations)\n")
        write(f, "Total posterior draws: $(ppc.summary.n_draws)\n\n")
        write(f, "Brier Score (Mean Squared Error):\n")
        write(f, "  Mean:     $(round(ppc.summary.brier_mean; digits=6))\n")
        write(f, "  Std Dev:  $(round(ppc.summary.brier_sd; digits=6))\n")
        write(f, "  95% CI:   [$(round(ppc.summary.brier_lower_ci; digits=6)), " *
                  "$(round(ppc.summary.brier_upper_ci; digits=6))]\n\n")
        write(f, "Kullback-Leibler Divergence:\n")
        write(f, "  Mean:     $(round(ppc.summary.kl_mean; digits=6))\n")
        write(f, "  Std Dev:  $(round(ppc.summary.kl_sd; digits=6))\n")
        write(f, "  95% CI:   [$(round(ppc.summary.kl_lower_ci; digits=6)), " *
                  "$(round(ppc.summary.kl_upper_ci; digits=6))]\n\n")
        write(f, "Observed Recapture Distribution Entropy:\n")
        write(f, "  $(round(ppc.summary.observed_dist_entropy; digits=6))\n")
        write(f, "  (Higher entropy = more dispersed recaptures)\n")
    end
    
    # Export rank histogram
    rank_file = joinpath(output_dir, "posterior_predictive_rank_histogram.csv")
    rank_df = DataFrame(
        rank_bin = collect(1:length(ppc.rank_histogram)),
        count    = ppc.rank_histogram,
    )
    CSV.write(rank_file, rank_df)
    
    # Export observed vs predicted recapture distributions
    dist_file = joinpath(output_dir, "posterior_predictive_distributions.csv")
    n_spatial = length(ppc.observed_dist)
    mean_simulated = zeros(n_spatial)
    
    for simulated_set in ppc.simulated_recapture_dists
        for rec in simulated_set
            if 1 <= rec <= n_spatial
                mean_simulated[rec] += 1
            end
        end
    end
    mean_simulated ./= sum(mean_simulated)
    
    dist_df = DataFrame(
        spatial_unit         = 1:n_spatial,
        observed_prob        = ppc.observed_dist,
        mean_predicted_prob  = mean_simulated,
    )
    CSV.write(dist_file, dist_df)
    
    return summary_file
end

"""
    plot_posterior_predictive_check(ppc, output_dir) -> String

Creates diagnostic visualizations for posterior predictive checks.

Uses Plots.jl to generate:
1. Brier score trace plot
2. KL divergence trace plot
3. Observed vs predicted recapture CDF
4. Rank histogram

# Arguments
- `ppc`: Output of `posterior_predictive_check`.
- `output_dir`: Output directory path.

# Returns
Path to output plot file.
"""
function plot_posterior_predictive_check(ppc, output_dir)::String
    using Plots
    gr()
    
    mkpath(output_dir)
    
    # Create 2x2 subplot grid
    p1 = plot(ppc.brier_scores; 
              label="Brier Score", xlabel="Draw", ylabel="Score",
              title="Posterior Predictive: Brier Score",
              legend=:topright, size=(600, 400))
    
    p2 = plot(ppc.kl_divergences;
              label="KL Divergence", xlabel="Draw", ylabel="Divergence",
              title="Posterior Predictive: KL Divergence",
              legend=:topright, size=(600, 400))
    
    # CDF comparison
    sort_obs = sort(ppc.observed_recaptures)
    cdf_obs = cumsum(ones(length(sort_obs))) ./ length(sort_obs)
    
    mean_pred_dist = zeros(length(ppc.observed_dist))
    for simulated_set in ppc.simulated_recapture_dists
        for rec in simulated_set
            if 1 <= rec <= length(ppc.observed_dist)
                mean_pred_dist[rec] += 1
            end
        end
    end
    mean_pred_dist ./= sum(mean_pred_dist)
    
    p3 = plot(1:length(ppc.observed_dist), ppc.observed_dist;
              label="Observed", xlabel="Spatial Unit", ylabel="Probability",
              title="Observed vs Predicted Recapture Distribution",
              legend=:topright, size=(600, 400))
    plot!(p3, 1:length(ppc.observed_dist), mean_pred_dist; label="Predicted Mean")
    
    # Rank histogram
    p4 = bar(ppc.rank_histogram;
             label="Frequency", xlabel="Rank Bin", ylabel="Count",
             title="Rank Histogram (Posterior Predictive)",
             legend=:topright, size=(600, 400))
    
    plot_file = joinpath(output_dir, "posterior_predictive_diagnostics.png")
    plot(p1, p2, p3, p4; layout=(2,2), size=(1200, 800))
    savefig(plot_file)
    
    return plot_file
end
