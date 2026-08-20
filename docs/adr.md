# Mechanistic Spatiotemporal Modeling: An Advection-Diffusion-Reaction (ADR) Framework

## 1. Introduction to Mechanistic Spatiotemporal Models

In ecological modeling, understanding how and why populations move and distribute themselves across a landscape is a fundamental challenge. Mechanistic spatiotemporal models provide a powerful, process-based framework for this task by explicitly representing the underlying ecological processes that drive population dynamics.

The Advection-Diffusion-Reaction (ADR) model is a cornerstone of this approach. It describes the change in population density over time and space as a result of three core processes:

1.  **Advection**: Directed, deterministic movement, typically along an environmental gradient. For example, snow crabs moving towards their preferred water depth or temperature.
2.  **Diffusion**: Random, undirected movement or dispersal. This captures the inherent stochasticity in individual behavior and exploratory movements.
3.  **Reaction**: Local population dynamics, such as births, deaths, and density-dependent competition, that are independent of movement. This is often modeled using logistic growth.

By simulating these core processes, the ADR framework allows us to test hypotheses about movement ecology and make predictions about population distributions under changing environmental conditions.

## 2. The Advection-Diffusion-Reaction (ADR) Mathematical Framework

### 2.1. Continuous Formulation (PDE)

Mathematically, a general ADR process is described by a partial differential equation (PDE) that governs the evolution of population density $N(\mathbf{s}, t)$ at a spatial location $\mathbf{s}$ and time $t$ (Okubo, 1980; Turchin, 1998):

$$
\frac{\partial N(\mathbf{s}, t)}{\partial t} = \underbrace{\nabla \cdot (D \nabla N(\mathbf{s}, t))}_{\text{Diffusion}} - \underbrace{\nabla \cdot (\mathbf{v} N(\mathbf{s}, t))}_{\text{Advection}} + \underbrace{f(N(\mathbf{s}, t))}_{\text{Reaction}}
$$

Where:
-   $\nabla \cdot$ is the divergence operator and $\nabla$ is the gradient operator.
-   $D$ is the **diffusion coefficient**, controlling the rate of random movement.
-   $\mathbf{v}$ is the **advection velocity field**, defining the direction and speed of directed movement.
-   $f(N)$ is the **reaction term**, representing local population growth (e.g., logistic growth $rN(1 - N/K)$).

Our goal is to implement a discrete version of this PDE within a Bayesian framework. This will allow us to estimate its key parameters ($D$, components of $\mathbf{v}$, etc.) by fitting the model to ecological data, such as population surveys and individual telemetry tracks (Hooten & Hefley, 2019).

### 2.2. Discretization for Bayesian Inference

To fit the model to data, we must discretize the continuous PDE. We represent the landscape as a set of connected spatial units (a graph), where each unit $s$ has a population density $N_s$. The change in density over a discrete time step evolves according to:

$$N_t = \mathbf{M}_{prop}^{-1} N_{t-1} + f(N_{t-1}) + \epsilon_t$$

Where:
-   $N_t$ is the vector of population densities across all spatial units at time $t$.
-   $\mathbf{M}_{prop}$ is the **propagator matrix**, which implicitly handles the movement (advection and diffusion) of the population over a time step.
-   $f(N_{t-1})$ is the vector of reaction terms (local growth) in each unit.
-   $\epsilon_t$ is a stochastic innovation term, accounting for process noise.

The propagator matrix is the mechanistic core of the model and is defined as:

$$\mathbf{M}_{prop} = (\mathbf{I} - S \cdot \mathbf{V} \cdot \mathbf{A} - D \cdot \mathbf{L})$$

Here:
-   $\mathbf{I}$ is the identity matrix.
-   $S$ is the advection strength parameter.
-   $\mathbf{V}$ is the habitat velocity field (derived from environmental gradients).
-   $\mathbf{A}$ is the **advection operator**, a matrix that approximates the gradient on the spatial graph.
-   $D$ is the diffusion coefficient.
-   $\mathbf{L}$ is the **graph Laplacian**, a matrix that approximates the diffusion process between connected spatial units.

This formulation uses an implicit time-stepping scheme, where the next state $N_t$ is found by solving a system of linear equations. This method is numerically stable and allows for larger time steps than explicit methods, which is crucial for long-term ecological simulations.

## 3. Modeling the Ecological Processes

### 3.1. Advection: Movement Along Habitat Gradients

Advection, or directed movement, is driven by an organism's preference for certain environmental conditions. We model this by first defining a Habitat Suitability Index (HSI). For a species like snow crab, HSI might be a function of water depth, with individuals preferring an optimal depth. A common choice is a Gaussian function:

$$ \text{HSI}(s) = \exp\left(-\frac{1}{2} \left(\frac{\text{depth}(s) - \text{depth}_{\text{opt}}}{\text{sd}_{\text{depth}}}\right)^2\right) $$

The advection velocity field, $\mathbf{v}$, is then defined as the gradient of this HSI field ($\mathbf{v} \propto \nabla \text{HSI}$). This means that individuals will tend to move "uphill" on the HSI landscape, from areas of lower suitability to areas of higher suitability. The parameter $S$ (advection strength) scales the magnitude of this directed movement.

### 3.2. Diffusion: Random Movement and the Graph Laplacian

Diffusion represents the random, undirected component of movement. On our discrete spatial graph, this process is modeled using the **graph Laplacian** matrix, $\mathbf{L}$. The Laplacian, constructed from the graph's adjacency and degree matrices, is the discrete analogue of the continuous Laplace operator and naturally describes the "flow" of a quantity (in this case, population density) between connected spatial units (Besag, 1974). The diffusion coefficient $D$ scales the overall rate of this random dispersal.

### 3.3. Reaction: Local Population Dynamics

The reaction term, $f(N)$, accounts for changes in population density due to local births and deaths, independent of movement. A standard choice for this term is the logistic growth model, which incorporates a maximum intrinsic growth rate ($r$) and a carrying capacity ($K$) for each spatial unit.

## 4. Data Integration in a Joint Likelihood Framework

A key strength of this framework is its ability to integrate different data types, such as broad-scale population surveys and fine-scale individual telemetry, into a single joint model (Hooten & Hefley, 2019). This approach, often called an Integrated Population Model (IPM), leverages the strengths of each data source to provide a more robust and comprehensive understanding of population dynamics (Johnson et al., 2008).

### 4.1. Likelihood for Population Density Data

Data from population surveys (e.g., trawl surveys) provide counts of individuals at specific locations and times. These counts, $C(s, t)$, can be linked to the model's predicted density, $N(s, t)$, through an observation model, typically a Poisson or Negative Binomial distribution:

$$ C(s, t) \sim \text{Poisson}(q \cdot N(s, t) \cdot \text{Area}(s)) $$

where $q$ is a catchability coefficient that accounts for survey gear efficiency.

### 4.2. Likelihood for Individual Telemetry Data

Telemetry data provides mark-recapture information for specific individuals. The probability of an individual moving from a release unit ($u_{rel}$) to a recapture unit ($u_{rec}$) is directly given by the elements of the inverted propagator matrix, which serves as a one-step transition matrix: $\mathbf{\Gamma} = \mathbf{M}_{prop}^{-1}$.

To account for heterogeneity among individuals (e.g., differences in size, sex, or condition), we can modify these transition probabilities based on individual-level covariates, $z_m$:

$$ P(u_{rec} | u_{rel}, z_m) = \frac{\mathbf{\Gamma}_{u_{rel}, u_{rec}}^{\exp(\beta_{het} z_m)}}{\sum_j \mathbf{\Gamma}_{u_{rel}, j}^{\exp(\beta_{het} z_m)}} $$

Here, $\beta_{het}$ is a parameter that captures the effect of the covariate $z_m$ on movement probability. The likelihood for the observed telemetry transitions is then formulated as a Categorical distribution. By combining the likelihoods for both the survey and telemetry data, we can create a joint model that leverages the strengths of both data sources to provide a more robust understanding of animal movement and population dynamics.

### 5. An example using simulated data

#### Setup

We use Julia and Turing to model.

```julia

using Pkg

for pk in ["Turing", "Distributions", "DataFrames", "Statistics", "LinearAlgebra", "Random", "Graphs", "SparseArrays"]
    if !haskey(Pkg.dependencies(), pk)
        Pkg.add(pk)
    end
end

using Turing, Distributions, DataFrames, Statistics, LinearAlgebra, Random, Graphs, SparseArrays

include("adr.jl")

```

#### Simulated data

Simulate mark-recapture positions on a grid of 1000 X 1000 km.

```julia
 ## Example

This example demonstrates a complete workflow for simulating data and fitting a joint Advection-Diffusion-Reaction (ADR) and telemetry model using the `bstm` framework. The model is specified using the unified `movement` component, which encapsulates both the population dynamics and the mark-recapture likelihood.

### 1. Setup and Data Simulation

First, we generate a synthetic dataset. This includes spatiotemporal population density data and telemetry data for marked individuals in a "long" format, which is a common and intuitive structure for such data.

```julia 

sim_data = generate_ADR_simulation_bundle(1000.0, 100, 5, 100) 

```

#### Joint model specification

Next, we define the joint model using the @bstm macro. The core of the model is the movement component, which handles the ADR dynamics and integrates the telemetry data. Priors for the advection velocity, diffusion coefficient, process noise, and individual heterogeneity are all specified directly within the movement call.

```julia
# The ADR process and telemetry likelihood are handled by the `movement` component.
# The telemetry data is passed as a DataFrame.
model_instance = @bstm(
    likelihood(density, family=:poisson) ~
        intercept(prior=Normal(0, 5)) +
        movement(unit_id, year,
            velocity = Normal(1.0, 0.5),
            diffusion = LogNormal(-1.0, 1.0),
            sigma = Exponential(1.0),
            beta_het = Normal(0.0, 0.5),
            mark_recapture_data = sim_data.telemetry_data
        ),
    sim_data.data,
    W = sim_data.au.W
)

# Sample from the posterior distribution using the NUTS sampler.
println("Starting posterior sampling...")
chain_results = sample(model_instance, NUTS(0.65), 500; progress=true)

println("\n--- Recovered Mechanistic Parameters ---")
display(chain_results)


```


#### Parameter estimation

MCMC and Diagnostic Analysis

```julia
  # Pre-compute the velocity field for post-hoc analysis
avg_p = [mean(sim_data.data.habitat_p[sim_data.data.unit_id .== i]) for i in 1:sim_data.n_spatial]
g_side = Int(sqrt(sim_data.n_spatial))
vel_vectors = compute_velocity_field(avg_p, g_side, 1.5, :exponential)

# Run the synthesis function to generate all diagnostic plots
synthesize_adr_results(chain_results, sim_data, vel_vectors)

```

#### Visualizations
 
Map Synthesis and Prediction visualization

```julia
 
if haskey(full_results.plots, :spatial_denoised)
    println("Visualizing Recovered Latent Intensity:")
    display(full_results.plots.spatial_denoised)
end

if haskey(full_results.plots, :temporal)
    println("Visualizing Recovered Population Trend:")
    display(full_results.plots.temporal)
end

# Extract and map a specific individual path from the posterior transition matrix
println("\nFinal Analysis: Assessed Joint Recovery of Abundance and Movement Dynamics.")


# Execution logic if results from previous cells are available
if @isdefined(model_joint_full) && @isdefined(joint_chain)
    dynamics_prediction = predict_and_plot_movement_dynamics(
        model_joint_full,
        joint_chain,
        telemetry_mat,
        au_context,
        data_df,
        target_mark_ids =
    )
else
    println("Model or Chain objects not found. Ensure sampling execution is complete.")
end


# Execution and Synthesis of Analytic Outputs
if @isdefined(dynamics_synthesis)
    println("--- Restoring Parity: Generating Regional Connectivity Matrix ---")

    # Define sample strata based on X-coordinate (e.g., East/West split at center of domain)
    cents_x = [c for c in au_context.centroids]
    strata = [x > 500.0 ? 2 : 1 for x in cents_x]

    # Extract posterior mean Gamma from previous propagator reconstruction (Gamma_Post from cell 16644d0d)
    if @isdefined(Gamma_Post)
        conn_mat = calculate_regional_connectivity(Gamma_Post, au_context, strata)

        plt_conn = heatmap(["West", "East"], ["West", "East"], conn_mat,
            title = "Regional Transfer Probability Matrix",
            xlabel = "To Region",
            ylabel = "From Region",
            color = :viridis,
            clims = (0, 1))
        display(plt_conn)
    else
        println("Warning: Gamma_Post not found. Ensure posterior synthesis has been executed.")
    end

    println("--- Restoring Parity: Generating A/D Ratio Histogram ---")
    plt_ad = plot_ad_ratio_distribution(
        dynamics_synthesis.advection_field,
        dynamics_synthesis.diffusion_field,
        au_context
    )
    display(plt_ad)

    println("--- Restoring Parity: Population Flow Vector Field ---")
    # Reconstructs the 'Advection Field (Net Flow)' quiver plot
    # Vectors indicate the mean direction and magnitude of habitat-coupled movement
    cents_y = [c for c in au_context.centroids]
    
    # Access velocity field components if available in synthesis, else default to derived advection
    # We scale the vectors for visualization clarity
    v_scale = 5.0
    plt_vectors = quiver(cents_x, cents_y,
        quiver = (dynamics_synthesis.advection_field .* v_scale, zeros(length(cents_x))),
        title = "Recovered Population Flow Vectors",
        aspect_ratio = :equal,
        color = :red,
        xlabel = "X (km)",
        ylabel = "Y (km)")
    display(plt_vectors)

else
    println("Reconstruction data (dynamics_synthesis) not found. Please ensure the synthesis step is complete.")
end


# Comparative Validation of Persistence Effects
if @isdefined(Gamma_Post) && @isdefined(au_context)
    println("--- Simulating Paths with High Temporal Autocorrelation (rho=2.0) ---")
    
    random_starts = rand(1:length(au_context.centroids), 3)
    n_sim_steps = 20
    
    # Generate paths with and without persistence
    paths_standard = simulate_posterior_trajectories(Gamma_Post, random_starts, n_sim_steps, au_context, rho_persistence=0.0)
    paths_persistent = simulate_posterior_trajectories(Gamma_Post, random_starts, n_sim_steps, au_context, rho_persistence=2.0)
    
    plt_comp = plot(layout=(1, 2), size=(1000, 500), aspect_ratio=:equal)
      
    render_paths!(plt_comp, paths_standard, "Standard Markovian Path")
    render_paths!(plt_comp, paths_persistent, "Persistent Path (rho=2.0)")
    
    display(plt_comp)
else
    println("State Requirement Error: Gamma_Post or au_context missing.")
end



# #
# Synthesis: Multi-Year Dynamic Projection
if @isdefined(population_density_matrix) && @isdefined(au_context)
    println("--- Initiating Multi-Year Mechanistic Path Projection ---")
    
    # For demo, we simulate a sequence of Gammas based on the estimated field drift
    # In a full model, these come from Gamma_Post at each time slice
    n_years_total = size(population_density_matrix, 2)
    
    # Generate a dummy Gamma sequence based on Propagator_Post (from cell 16644d0d)
    # In practice, this uses year-specific habitat velocity fields
    gamma_seq = [Gamma_Post for _ in 1:n_years_total]
    
    # Select 5 individuals starting in high-density areas at Year 1
    high_density_units = sortperm(population_density_matrix[:, 1], rev=true)[1:10]
    starts = rand(high_density_units, 5)
    
    # Simulate paths across the entire 5-year study period
    dynamic_paths = simulate_mechanistic_trajectories(
        gamma_seq, 
        starts, 
        1, 
        au_context, 
        rho_persistence=1.5, 
        n_years_sim=4
    )
    
    plt_dyn = plot(aspect_ratio=:equal, title="Mechanistic 5-Year Path Projections", legend=:outerright)
    
    # Render Background Polygons
    for poly in au_context.polygons
        if length(poly) > 2
            plot!(plt_dyn, [pt for pt in poly], [pt for pt in poly], fillalpha=0.02, color=:black, lw=0.1, label=nothing)
        end
    end
    
    # Render Individual Trajectories
    for i in 1:size(dynamic_paths, 1)
        idx_list = dynamic_paths[i, :]
        px = [au_context.centroids[id] for id in idx_list]
        py = [au_context.centroids[id] for id in idx_list]
        plot!(plt_dyn, px, py, marker=:circle, markersize=2.0, lw=1.5, label="Indiv $i")
    end
    
    display(plt_dyn)
else
    println("Incomplete Context: population_density_matrix or au_context missing.")
end



# # 
# Integrated Synthesis: Multi-Year Raster-Driven Trajectories
# Rationale: We demonstrate the coupling of a dynamic suitability field (e.g. from an SST raster) to individual paths.
if @isdefined(sim_bundle) && @isdefined(au_context)
    println("--- Generating Multi-Year Transition Sequence from Suitability ---")
    
    n_years = sim_bundle.n_years
    n_spatial = sim_bundle.n_spatial
    
    # Container for the sequence of matrices Gamma_t
    dynamic_gamma_seq = Vector{SparseMatrixCSC{Float64, Int}}(undef, n_years)
    
    for t in 1:n_years
        # Extract suitability vector for Year t from the simulation data
        # In practice, this would be sampled from a Raster object
        year_mask = sim_bundle.data.year .== t
        s_vec = sim_bundle.data.habitat_p[year_mask]
        
        # Compute year-specific kernel
        # Sensitivity (1.5) determines how aggressively individuals follow the gradient
        dynamic_gamma_seq[t] = compute_suitability_transition_kernel(
            s_vec, 
            au_context.W, 
            1.5, 
            0.05
        )
    end
    
    # Simulate 5 individual paths using the dynamic sequence
    # Rationale: Individuals now navigate an environment where connectivity changes every year
    starts = rand(1:n_spatial, 5)
    raster_driven_paths = simulate_mechanistic_trajectories(
        dynamic_gamma_seq, 
        starts, 
        1, 
        au_context, 
        rho_persistence=1.2, 
        n_years_sim = n_years - 1
    )
    
    # Visualization
    plt_raster = plot(aspect_ratio=:equal, title="Raster-Driven Mechanistic Trajectories", legend=:outerright)
    
    # Background: Mean Suitability Field
    avg_s = [mean(sim_bundle.data.habitat_p[sim_bundle.data.unit_id .== i]) for i in 1:n_spatial]
    for (i, poly) in enumerate(au_context.polygons)
        if length(poly) > 2
            px = [p for p in poly]; py = [p for p in poly]
            plot!(plt_raster, px, py, fill=(true, palette(:viridis)[Int(round(avg_s[i]*255)+1)]), fillalpha=0.3, lw=0.1, label=nothing)
        end
    end
    
    # Trajectories
    for i in 1:size(raster_driven_paths, 1)
        idx_list = raster_driven_paths[i, :]
        px = [au_context.centroids[id] for id in idx_list]
        py = [au_context.centroids[id] for id in idx_list]
        plot!(plt_raster, px, py, marker=:circle, markersize=2.5, lw=2.0, label="Path $i")
    end
    
    display(plt_raster)
else
    println("Missing Context: sim_bundle or au_context required for demonstration.")
end

```

### 6. An example using snow crab data



## 7. Assumptions, Weaknesses, and Extensions

While powerful, the ADR framework rests on several key assumptions and has inherent weaknesses that are important to acknowledge.

### 7.1. Model Assumptions
*   **Continuum Assumption (PDE):** The original PDE formulation assumes that population density is a continuous and differentiable field. This is an idealization, as populations consist of discrete individuals.
*   **Discretization Scheme:** The discrete model's accuracy depends on the chosen spatial and temporal resolution. The graph representation (areal units) imposes a fixed structure on movement, which can be a source of bias if the units do not align with ecological realities (i.e., the Modifiable Areal Unit Problem, MAUP).
*   **Homogeneous Processes:** The basic model often assumes that parameters like diffusion ($D$), advection strength ($S$), and growth rates ($r, K$) are constant across space and time. This is a major simplification, as animal behavior and population dynamics are often state-dependent and vary with environmental conditions.
*   **Gradient-Following Advection:** The assumption that directed movement follows the gradient of an HSI is a common but simplified model of behavior. It does not account for memory, social interactions, or complex navigational strategies.
*   **Fickian Diffusion:** The diffusion term models movement as a random walk, which may not accurately capture more complex dispersal patterns like Lévy flights or correlated random walks.

### 7.2. Weaknesses and Limitations
*   **Parameter Identifiability:** In complex models, it can be difficult to separately identify the effects of advection, diffusion, and local growth, especially with limited data. For example, a high mortality rate in an area (reaction) can be difficult to distinguish from strong net emigration (advection/diffusion).
*   **Computational Cost:** While the implicit scheme is stable, solving the linear system involving the inverse of the propagator matrix ($\mathbf{M}_{prop}^{-1}$) can be computationally intensive for very large graphs (many spatial units).
*   **Static Environment:** The basic framework assumes a static HSI and therefore a static advection field. In reality, environmental conditions change, which would require making the propagator matrix $\mathbf{M}_{prop}$ time-dependent.

### 7.3. Potential Extensions
The ADR framework is highly flexible and can be extended to address these limitations:
*   **Spatially-Varying Coefficients:** Allow parameters like $D$ and $S$ to vary across the landscape as a function of environmental covariates (e.g., making diffusion higher in open habitats).
*   **Dynamic Environments:** Incorporate time-varying environmental data (e.g., sea surface temperature) to create a time-varying HSI and advection field.
*   **Non-Fickian Diffusion:** Replace the simple diffusion term with more complex operators to model different types of random movement.
*   **Structured Population Models:** Extend the reaction term to include age or size structure, moving from a simple logistic growth model to a more realistic matrix population model.

## 8. References

Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192-225.

Hooten, M. B., & Hefley, T. J. (2019). *Bringing Bayesian models to life: A guide to fitting, testing, and interpreting models of ecological dynamics*. CRC Press.

Johnson, D. S., Laake, J. L., & Ver Hoef, J. M. (2008). A model-based approach for making ecological inference from distance sampling data. *The Journal of Wildlife Management*, 72(6), 1387-1393.

Okubo, A. (1980). *Diffusion and Ecological Problems: Mathematical Models*. Springer-Verlag.

Turchin, P. (1998). *Quantitative analysis of movement: measuring and modeling population redistribution in animals and plants*. Sinauer Associates.



## Example

```julia

# Rationale: Main execution block to run the entire ADR simulation and analysis workflow 
# Execute the main workflow

main()


# 1. Environment and Spatial Context Establishment
# Generate a coordinate set to define the irregular spatial units
domain_extent = 1000.0
initial_points_x = rand(Uniform(0, domain_extent), 500)
initial_points_y = rand(Uniform(0, domain_extent), 500)

# Partition the domain into discrete areal units using CVT
# Rationale: This creates the 'au_context' required for all spatial plotting functions
au_context = assign_spatial_units(
    initial_points_x,
    initial_points_y,
    area_method = :cvt,
    target_units = 100,
    buffer_dist = 0.0
)

# 2. Environmental Field derivation
# Construct a suitability grid and map it to the advection velocity vectors
grid_size = Int(sqrt(length(au_context.centroids)))
habitat_probability_grid = reshape(zeros(length(au_context.centroids)), grid_size, grid_size)

# Simulate a central suitability peak
for i in 1:length(au_context.centroids)
    cx, cy = au_context.centroids[i]
    dist_center = sqrt((cx - 500.0)^2 + (cy - 500.0)^2)
    habitat_probability_grid[i] = exp(-dist_center / 250.0)
end

# Map habitat gradients to velocity components using exponential coupling
velocity_field_components = map_habitat_to_advection_velocity(
    habitat_probability_grid,
    1.5,           # Advection strength
    0.05,          # Threshold
    1.0,           # Scale
    :exponential   # Relative gradient coupling
)

# 3. Model State and Result Synthesis
# Rationale: If a model and chain exist, we perform the unified prediction and plotting.
# We utilize @isdefined to ensure script robustness within the notebook execution flow.

if @isdefined(model_joint_full) && @isdefined(joint_chain)
    println("--- Initiating Movement Dynamics Visualization Suite ---")
    
    # Prepare telemetry data for visualization (using indices from the simulation bundle)
    # Format: [Release_U, Recapture_U, T_Rel, T_Rec, Covariate]
    current_telemetry_matrix = telemetry_mat 
    
    # Invoke the integrated prediction and plotting engine
    # target_mark_ids specifies which individuals to generate track maps for
    dynamics_synthesis = predict_and_plot_movement_dynamics(
        model_joint_full,
        joint_chain,
        current_telemetry_matrix,
        au_context,
        data_df,
        target_mark_ids =
    )
    
    # 4. Supplemental Diagnostic Plots
    # We can explicitly access components of the synthesis bundle for custom layouts
    
    # Visualize the magnitude of the advection field across the polygons
    adv_mag_plot = plot_choropleth(
        dynamics_synthesis.advection_field,
        au_context.polygons,
        title = "Synthesis: Posterior Advection Field Magnitude",
        cmap = :viridis
    )
    
    # Visualize the diffusion field
    diff_coeff_plot = plot_choropleth(
        dynamics_synthesis.diffusion_field,
        au_context.polygons,
        title = "Synthesis: Posterior Diffusion Coefficient",
        cmap = :plasma
    )
    
    # Display supplemental diagnostics
    display(adv_mag_plot)
    display(diff_coeff_plot)
    
    println("--- Visualization Suite Execution Complete ---")
else
    # Fallback logic if the model has not been sampled
    println("Model or sampling chain not detected. Generating prior-based habitat map only.")
    
    prior_habitat_plot = plot_choropleth(
        vec(habitat_probability_grid),
        au_context.polygons,
        title = "Prior Habitat Suitability Surface",
        cmap = :terrain
    )
    
    display(prior_habitat_plot)
end


```