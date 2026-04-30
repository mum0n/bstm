
function example_data( datatype, N=200, N_inducing=10; period=12.0, seed=42 )
 
  Random.seed!(seed)


  if datatype == "scottish_lip_cancer_data"
  
      # data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

      # y: the observed lip cancer case counts on a per-county basis
      # x: an area-specific continuous covariate that represents the proportion of the population employed in agriculture, fishing, or forestry (AFF)
      # E: the expected number of cases, used as an offset,
      # adj: a list of region ids for adjacent regions
      # num: a list of the number of neighbors for each region

      N = 56

      y   = [ 9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
      16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
      10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0]
      
      E = [1.4, 8.7, 3.0, 2.5, 4.3, 2.4, 8.1, 2.3, 2.0, 6.6, 4.4, 1.8, 1.1, 3.3, 7.8, 4.6,
      1.1, 4.2, 5.5, 4.4, 10.5,22.7, 8.8, 5.6,15.5,12.5, 6.0, 9.0,14.4,10.2, 4.8, 2.9, 7.0,
      8.5, 12.3, 10.1, 12.7, 9.4, 7.2, 5.3,  18.8,15.8, 4.3,14.6,50.7, 8.2, 5.6, 9.3, 88.7, 
      19.6, 3.4, 3.6, 5.7, 7.0, 4.2, 1.8]
      
      x = [16,16,10,24,10,24,10, 7, 7,16, 7,16,10,24, 7,16,10, 7, 7,10,
      7,16,10, 7, 1, 1, 7, 7,10,10, 7,24,10, 7, 7, 0,10, 1,16, 0, 
      1,16,16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1,16,10]
      
      # fake groups
      groups = [1, 1, 1,1 ,1,1,1, 1, 1,1,1,1,1,2,2,2,2, 2, 2,1,
      3,3,3, 1, 1, 1, 1, 1,1,1, 1,1,1, 1, 1, 1,1, 1,1, 1,
      1,1,2, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1,3,2]
      
      
      adj = [ 5, 9,11,19, 7,10, 6,12, 18,20,28, 1,11,12,13,19,
      3, 8, 2,10,13,16,17, 6, 1,11,17,19,23,29, 2, 7,16,22, 1, 5, 9,12,
      3, 5,11, 5, 7,17,19, 31,32,35, 25,29,50, 7,10,17,21,22,29,
      7, 9,13,16,19,29, 4,20,28,33,55,56, 1, 5, 9,13,17, 4,18,55,
      16,29,50, 10,16, 9,29,34,36,37,39, 27,30,31,44,47,48,55,56,
      15,26,29, 25,29,42,43, 24,31,32,55, 4,18,33,45, 9,15,16,17,21,23,25,
      26,34,43,50, 24,38,42,44,45,56, 14,24,27,32,35,46,47, 14,27,31,35,
      18,28,45,56, 23,29,39,40,42,43,51,52,54, 14,31,32,37,46,
      23,37,39,41, 23,35,36,41,46, 30,42,44,49,51,54, 23,34,36,40,41,
      34,39,41,49,52, 36,37,39,40,46,49,53, 26,30,34,38,43,51, 26,29,34,42,
      24,30,38,48,49, 28,30,33,56, 31,35,37,41,47,53, 24,31,46,48,49,53,
      24,44,47,49, 38,40,41,44,47,48,52,53,54, 15,21,29, 34,38,42,54,
      34,40,49,54, 41,46,47,49, 34,38,49,51,52, 18,20,24,27,56,
      18,24,30,33,45,55]
              
      num = [4, 2, 2, 3, 5, 2, 5, 1,  6,  4, 4, 3, 4, 3, 3, 6, 6, 6 ,5, 
      3, 3, 2, 6, 8, 3, 4, 4, 4,11,  6, 7, 4, 4, 9, 5, 4, 5, 6, 5, 
      5, 7, 6, 4, 5, 4, 6, 6, 4, 9, 3, 4, 4, 4, 5, 5, 6]
      
      # areal unit id of data (y) 
      auid = 1:N  # simple here as 1:1 au:data correspondence
      nAU = N
      
      N_edges = Integer( length(adj) / 2 );
      node1 =  fill(0, N_edges); 
      node2 =  fill(0, N_edges); 
      i_adjacency = 0;
      i_edge = 0;
      for i in 1:N
      for j in 1:num[i]
          i_adjacency = i_adjacency + 1;
          if i < adj[i_adjacency]
              i_edge = i_edge + 1;
              node1[i_edge] = i;
              node2[i_edge] = adj[i_adjacency];
          end
      end
      end
      
      e = Edge.(node1, node2)
      g = Graph(e)
      W = adjacency_matrix(g)
      
      # scaling_factor = scaling_factor_bym2(W)  # 0.22585017121540601 
      D = diagm(vec( sum(W, dims=2) ))
      
      # x = collect(x)
      x_scaled = (x .- mean(x)) ./ std(x)
      X = DataFrame( Intercept=ones( N ), x=x_scaled )
      X = Matrix(X)
      
      nX = size(X)[2]
      log_offset = log.(E)

      return D, W, X, log_offset, y, nX, nAU, node1, node2 # , scaling_factor 

  end


  if datatype=="scottish_lip_cancer_data_spacetime"
      # "expand" scottish lip cancer data  to a space-time version 
      # original data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

      # Base Spatial Data for 56 Counties
    
    # Load base spatial data
 # Base Spatial Data for 56 Counties
    # data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    nAU = 56

    y_base = [ 9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
    16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
    10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0]

    E_base = [1.4, 8.7, 3.0, 2.5, 4.3, 2.4, 8.1, 2.3, 2.0, 6.6, 4.4, 1.8, 1.1, 3.3, 7.8, 4.6,
    1.1, 4.2, 5.5, 4.4, 10.5,22.7, 8.8, 5.6,15.5,12.5, 6.0, 9.0,14.4,10.2, 4.8, 2.9, 7.0,
    8.5, 12.3, 10.1, 12.7, 9.4, 7.2, 5.3,  18.8,15.8, 4.3,14.6,50.7, 8.2, 5.6, 9.3, 88.7,
    19.6, 3.4, 3.6, 5.7, 7.0, 4.2, 1.8]

    x_base = [16,16,10,24,10,24,10, 7, 7,16, 7,16,10,24, 7,16,10, 7, 7,10,
    7,16,10, 7, 1, 1, 7, 7,10,10, 7,24,10, 7, 7, 0,10, 1,16, 0,
    1,16,16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1,16,10]

    adjacency = [ 5, 9,11,19, 7,10, 6,12, 18,20,28, 1,11,12,13,19,
    3, 8, 2,10,13,16,17, 6, 1,11,17,19,23,29, 2, 7,16,22, 1, 5, 9,12,
    3, 5,11, 5, 7,17,19, 31,32,35, 25,29,50, 7,10,17,21,22,29,
    7, 9,13,16,19,29, 4,20,28,33,55,56, 1, 5, 9,13,17, 4,18,55,
    16,29,50, 10,16, 9,29,34,36,37,39, 27,30,31,44,47,48,55,56,
    15,26,29, 25,29,42,43, 24,31,32,55, 4,18,33,45, 9,15,16,17,21,23,25,
    26,34,43,50, 24,38,42,44,45,56, 14,24,27,32,35,46,47, 14,27,31,35,
    18,28,45,56, 23,29,39,40,42,43,51,52,54, 14,31,32,37,46,
    23,37,39,41, 23,35,36,41,46, 30,42,44,49,51,54, 23,34,36,40,41,
    34,39,41,49,52, 36,37,39,40,46,49,53, 26,30,34,38,43,51, 26,29,34,42,
    24,30,38,48,49, 28,30,33,56, 31,35,37,41,47,53, 24,31,46,48,49,53,
    24,44,47,49, 38,40,41,44,47,48,52,53,54, 15,21,29, 34,38,42,54,
    34,40,49,54, 41,46,47,49, 34,38,49,51,52, 18,20,24,27,56,
    18,24,30,33,45,55]

    number_neighbours = [4, 2, 2, 3, 5, 2, 5, 1,  6,  4, 4, 3, 4, 3, 3, 6, 6, 6 ,5,
    3, 3, 2, 6, 8, 3, 4, 4, 4,11,  6, 7, 4, 4, 9, 5, 4, 5, 6, 5,
    5, 7, 6, 4, 5, 4, 6, 6, 4, 9, 3, 4, 4, 4, 5, 5, 6]
 
    # Build graph from adjacency info

    N_edges = Integer(length(adjacency) / 2)
    node1 = fill(0, N_edges)
    node2 = fill(0, N_edges)
    i_adjacency = 0
    i_edge = 0
    for i in 1:nAU
        for j in 1:number_neighbours[i]
            i_adjacency += 1
            if i < adjacency[i_adjacency]
                i_edge += 1
                node1[i_edge] = i
                node2[i_edge] = adjacency[i_adjacency]
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = adjacency_matrix(g)
    D = diagm(vec(sum(W, dims=2)))
 
    au = assign_spatial_units( W ) # "infer" from the adjacency network (W)
    pts_base = au.centroids
    
    N_total = nAU * n_years

    # 1. Random Walk Trend
    rw_trend = cumsum(randn(n_years) .* 0.5)

    # 2. Expand Data Vectors
    y_expanded = repeat(y_base, n_years)
    E_expanded = repeat(E_base, n_years)
    x_expanded = repeat(x_base, n_years)
    time_idx = repeat(1:n_years, inner=nAU)
    pts = repeat(pts_base, n_years)
    # The area_idx is the spatial unit identifier (1 to 56)
    area_idx = repeat(1:nAU, n_years)
 
    # 3. Add Random Walk + Noise to Response
    # Broadcast rw_trend across years
    trend_component = repeat(rw_trend, inner=nAU)
    noise = randn(N_total) .* 0.2

    # Final response: base_y + trend + noise (ensuring positive counts)
    y_final = floor.(Int, abs.(y_expanded .+ trend_component .+ noise))

    # 4. Final covariate matrix and offsets
    x_scaled = (x_expanded .- mean(x_expanded)) ./ std(x_expanded)
    X = Matrix(DataFrame(AFF=x_scaled))
    log_offset = log.(E_expanded)
   
    return (
        y=y_final, X=X, log_offset=log_offset, time_idx=time_idx,
        area_idx=area_idx, n_years=n_years, pts=pts, W=W, au=au
    )
    
  end

 

  if datatype == "regression"
    
    # make random data for analysis

    # NOTE: utility in terms of creating model matrix using schemas, etc
    
    # y data is a sine function with period 2pi and noise .. so phi should be close to 2pi or less
    # alpha is coefficient of covariate1 (beta5 in model output), 
    # cov2lev are levels of categorical covariate2 (beta1:4 in model output)
    # covar3 is integer from 1:10 . used as "time" / year for ar1 process
 
    N  = 500
    cov2lev = ("1"=>1, "2"=>1.25, "3"=>2, "4"=>1.5)
    alpha=0.1 
    
    xvar = vec(randn(N)*3.0  )
    
    data = DataFrame(
        xvar = xvar,
        covar1 = string.(rand( [1, 2, 3, 4], N)  ),  # factorial
        covar2 = vec(randn(N)),
        covar3 = vec(trunc.( Int, randn(N)*3 ) )
    )

    cov2 = replace(data.covar1, cov2lev[1], cov2lev[2], cov2lev[3], cov2lev[4] ) 
    data.yvar = sin.(vec(xvar)) + data.covar2 .* alpha .+ rand.(Normal.(cov2, 0.1))
    schm = StatsModels.schema(data, Dict(:covar1 => EffectsCoding()) )
    dm = StatsModels.apply_schema(StatsModels.term.((:xvar, :covar1, :covar2)), schm ) 
    modelcols = StatsModels.modelcols(StatsModels.MatrixTerm( dm ), data)
    coefnames = StatsModels.coefnames( dm )   # coef names of design matrix 
    termnames = StatsModels.termnames( dm)  # coef names of model data
    
    # alternative access:
    # fm = StatsModels.@formula( yvar ~ 1 + xvar + covar1 + covar2)
    # resp = response(fm, data)  # response
    # cols = modelcols(z, data)
    # o = reduce(hcat, cols) 
 
    return data, modelcols, coefnames, termnames, cov2lev

  end


  if datatype == "nonlinear"

    Xlatent = -10:0.01:10
    Xobs = -10:0.01:10 
    
    # function that describes latent process
    Y(x) = (x + 4) * (x + 1) * (x - 1) * (x - 3)  
    Ylatent = Y.(Xlatent)

    # "Observations" with noise
    Yobs = Y.(Xobs) .+ rand(Uniform(-100, 100), size(Xobs,1))
    
    return Xlatent, Ylatent, Xobs, Yobs

  end

 

  if datatype == "spatiotemporal_testdata"
     
    # A seasonal trend (Time) and a geographic hotspot (Space).
    # X_st (Inputs): A 3 x N matrix where Row 1 is Time, and Rows 2-3 are Space (Lat/Lon).
    # y (Observations): The noisy signal values.
    
    # Time: Continuous (0.0 to 10.0)
    # Space: Continuous Lat/Lon (-5.0 to 5.0)
    
    raw_time = rand(Uniform(0, 10), N)
    raw_space = rand(Uniform(-5, 5), 2, N) # 2 rows (Lat, Lon)
    
    # Signal = Sinusoidal Time + Gaussian Spatial Blob

    # Calculate true Y without noise
    y_true = Float64[]
    for i in 1:N

      t = raw_time[i]
      lat = raw_space[1,i]
      lon = raw_space[2,i]

      # Time component: Sine wave
      f_t = 2.0 * sin(t)
      
      # Spatial component: A "hotspot" at (0,0)
      dist_sq = lat^2 + lon^2
      f_s = 3.0 * exp(-dist_sq / 4.0) 
       
      y_true = push!(y_true, f_t + f_s)
    end


    # Add observational noise (Sigma = 0.3)
    y = y_true .+ 0.3 .* randn(N)
    
    # Format Inputs for ApproximateGPs
    # The model expects ColVecs (Column Vectors) wrapper for the kernel computations.
    
    # Full Data Inputs
    X_time = ColVecs(reshape(raw_time, 1, :)) # 1xN Matrix wrapped
    X_space = ColVecs(raw_space)              # 2xN Matrix wrapped
    
    # X_st (Space-Time) is 3xN: [Time; Lat; Lon]
    raw_st = vcat(reshape(raw_time, 1, :), raw_space)
    X_st = ColVecs(raw_st)
    
    # inducing points, consider using K-Means for real work
    idx_inducing = randperm(N)[1:N_inducing]
    inducing_matrix = raw_st[:, idx_inducing] # 3xM Matrix
    inducing_locs_st = ColVecs(inducing_matrix)
      
    return y, X_space, X_time, X_st, inducing_locs_st
 
  end

 

  if datatype =="spatiotemporal_icar"
    
    # Space and spatiotemporal model (ICAR)
     
    # W: 25x25 sparse adjacency matrix representing the grid neighbors. 
    
    #### Preliminaries: generate neighbourhood data 
      
    N_times = 10  # Number of time points
    N_r = 5  # n rows of spatial grid
    N_c = 5  # n cols of spatial grid
    N_areas = N_c * N_r
    N = N_areas * N_times
     
    # Function to create a 5x5 grid of polygons and return adjacency matrix
    
    W = lattice_adjacency_matrix(N_r, N_c)
 
    # Simulate AR(1) Term (a) ---
    ρ_a = 0.8
    σ_a = 0.5
    
    a = zeros(N_times)
    a[1] = rand(Normal(0, σ_a))
    for t in 2:N_times
        a[t] = ρ_a * a[t-1] + rand(Normal(0, σ_a * sqrt(1 - ρ_a^2)))
    end
    
    # Simulate BYM2 Spatial Term (u) ---
    # We use a simple ICAR simulation: u ~ MVN(0, inv(Q))
    D = Diagonal(sum(W, dims=2)[:])
    Q_spatial = D - W + 1e-6I # Add jitter for invertibility
    u_struc = rand(MvNormal(zeros(N_areas), inv(Hermitian(collect(Q_spatial)))))
    u_iid = rand(Normal(0, 1), N_areas)
    
    ϕ_u = 0.7  # 70% structured spatial effect
    σ_u = 0.3
    u_combined = σ_u .* (sqrt(ϕ_u) .* u_struc .+ sqrt(1 - ϕ_u) .* u_iid)
    
    # Simulate Spatio-Temporal Interaction (v) ---
    ρ_v = 0.5
    v = [rand(Normal(0, 0.2), N_areas) for _ in 1:N_times]
    for t in 2:N_times
        v[t] = ρ_v .* v[t-1] .+ rand(Normal(0, 0.1), N_areas)
    end
    
    # Map observations to indices
        
    # y: Vector of length 250 containing the noisy observations.
    
    # a_idx, u_idx, group_idx: Integer vectors mapping each observation to its respective time, area, and group.
    
    u_idx = repeat(1:N_areas, inner=N_times)
    a_idx = repeat(1:N_times, outer=N_areas)
    group_idx = a_idx # Same as time index
    
    y = Float64[]
    for i in 1:N
        η = a[a_idx[i]] + u_combined[u_idx[i]] + v[group_idx[i]][u_idx[i]]
        push!(y, η + rand(Normal(0, 0.1))) # Add observation noise
    end
    
    # println("Generated $(length(y)) observations for $N_areas areas over $N_times time steps.")

    return y, a_idx, u_idx, group_idx, W, N_times, N_areas, N

  end


  if datatype=="correlated_data"
  
    n = 200
    cormat = random_correlation_matrix(3, 0.1)
    o = rand( MvNormal( cormat^2 ), n )
    X = o[1,:]
    Y = o[2,:]
    Z = o[3,:]
    x = (X .- mean(X)) ./ std(X)
    y = (Y .- mean(Y)) ./ std(Y)
    z = (Z .- mean(Z)) ./ std(Z)
    
    # add categorical "data" (factors)
    nu = 5
    u = sample(1:nu, length(x), replace=true)  # categorical index
        
    return x,y,z,u,X,Y,Z  
  end

 
  if datatype == "PalmerPenguins"
    
    o = DataFrame(PalmerPenguins.load())

# names(o)
# 7-element Vector{String}:
#  "species"
#  "island"
#  "bill_length_mm"
#  "bill_depth_mm"
#  "flipper_length_mm"
#  "body_mass_g"
#  "sex"

    X = o[!,:bill_length_mm]
    Y = o[!,:flipper_length_mm]
    Z = o[!,:body_mass_g]
    n = size(o)[1]

    x = (X .- mean(X)) ./ std(X)
    y = (Y .- mean(Y)) ./ std(Y)
    z = (Z .- mean(Z)) ./ std(Z)
    
    # categorical "data" (factors)
    u = o[!,:island]
    
    return x,y,z,u,X,Y,Z  
  
  end

  
  if datatype == "PalmerPenguins_pca"
    
    o = DataFrame(PalmerPenguins.load())

# names(o)
# 7-element Vector{String}:
#  "species"
#  "island"
#  "bill_length_mm"
#  "bill_depth_mm"
#  "flipper_length_mm"
#  "body_mass_g"
#  "sex"
  
    sps = o[!,:species]
 
    X = o[:, [:bill_length_mm, :bill_depth_mm, :flipper_length_mm, :body_mass_g] ]
    vn = names(X)
    
    X = Matrix(X)
    nData = size(X, 1)

    X = X .- mean(X, dims=1)  # center

    # fake covariates
    G = zeros(nData, 2)
    G[:,1] = rand(Poisson(10), nData)
    G[:,2] = rand(Poisson(20), nData)
    
    id = recode(unwrap.( sps ), "Adelie"=>1, "Gentoo"=>2, "Chinstrap"=>3)
     
    return id, sps, X, G, vn 
  end

    
  if datatype == "PalmerPenguins_pca_nonlinear"
    
    id, sps, X, G, vn = example_data("PalmerPenguins")
  
    # non-linearize data to demonstrate ability of GPs to deal with non-linearity
    X[:, 1] = 0.5 * X[:, 1] .^ 2 + 0.1 * X[:, 1] .^ 3
    X[:, 2] = X[:, 2] .^ 3 + 0.2 * X[:, 2] .^ 4
    X[:, 3] = 0.1 * exp.(X[:, 3]) - 0.2 * X[:, 3] .^ 2
    X[:, 4] = 0.5 * (X[:, 4]).^ 2 + 0.01 * X[:, 4].^ 5
    
    # center
    X = X .- mean(X, dims=1)
    
    return id, sps, X, G, vn 
  end


  if datatype =="classification_testdata"
        
    num_samples = 300
    num_predictors = 2
    num_classes = 3 
    
    # Predictor variables
    x1 = randn(num_samples) # Predictor 1
    x2 = randn(num_samples) # Predictor 2

    # Design matrix (including intercept)
    X = hcat(ones(num_samples), x1, x2)

    # True coefficients for each class. One class is typically used as a reference (e.g., class 1 is all zeros).
    # The coefficients represent the linear predictor for each class relative to the reference.
    # Size: (num_predictors + 1) x (num_classes - 1)

    # True coefficients for Class 2 vs Class 1 (reference)
    true_beta_class2 = [0.5, 1.0, -0.5] # Intercept, x1_coeff, x2_coeff

    # True coefficients for Class 3 vs Class 1 (reference)
    true_beta_class3 = [-0.8, -0.7, 1.2] # Intercept, x1_coeff, x2_coeff

    # Combine into a matrix of coefficients, where the first column is for the reference class (all zeros)
    true_betas_all = hcat(zeros(num_predictors + 1), true_beta_class2, true_beta_class3)

    # Calculate the linear predictors for each class for each observation
    linear_predictors = X * true_betas_all # num_samples x num_classes

    # Convert linear predictors to probabilities for each class using softmax
    # StatsAPI.softmax operates on columns by default, so we'll transpose for row-wise operation
    probabilities_per_class = permutedims(StatsAPI.softmax(permutedims(linear_predictors)))

    # Generate class labels (y) based on these probabilities
    y = Vector{Int}(undef, num_samples)
    for i in 1:num_samples
        # Categorical(p) expects probabilities for each category
        y[i] = rand(Categorical(probabilities_per_class[i, :]))
    end

    println("Synthetic Softmax Regression Data Generated. First 10 observations (X1, X2, Probabilities for classes 1,2,3, Assigned Class):")
    display(hcat(round.(x1[1:10], digits=2), round.(x2[1:10], digits=2), round.(probabilities_per_class[1:10,:], digits=2), y[1:10]))

    println("\nCounts per class: ", StatsBase.countmap(y))

    return X, y, num_classes, probabilities_per_class, true_beta_class2, true_beta_class3

  end


  if datatype =="cluster_testdata"

    num_samples = 200
    num_clusters = 2

    # Parameters for data types:
    # Continuous: 1 dimension
    # Binomial: 1 dimension (e.g., number of successes out of 10 trials)
    # Multinomial: 1 dimension (e.g., counts in 3 categories)

    # True cluster parameters
    # Cluster 1
    μ_true_1 = [2.0]     # Continuous mean
    p_true_1 = 0.2      # Binomial success probability
    θ_true_1 = [0.6, 0.3, 0.1] # Multinomial probabilities for 3 categories

    # Cluster 2
    μ_true_2 = [7.0]
    p_true_2 = 0.8
    θ_true_2 = [0.1, 0.2, 0.7]

    # Combine into arrays for the model
    true_μ = [μ_true_1, μ_true_2]
    true_p = [p_true_1, p_true_2]
    true_θ = [θ_true_1, θ_true_2]

    true_π = [0.4, 0.6] # Mixing proportions

    # Binomial trials (fixed for all points)
    binomial_trials = 10
    # Multinomial categories (fixed)
    multinomial_categories = 3
    # Total count for multinomial (fixed for all points)
    multinomial_total_count = 20

    # Store generated data
    data_continuous = Vector{Float64}(undef, num_samples)
    data_binomial = Vector{Int}(undef, num_samples)
    data_multinomial = Matrix{Int}(undef, num_samples, multinomial_categories)
    cluster_assignments_true = Vector{Int}(undef, num_samples)

    for i in 1:num_samples
        # Assign cluster
        k = rand(Categorical(true_π))
        cluster_assignments_true[i] = k

        # Generate continuous data
        data_continuous[i] = rand(Normal(true_μ[k][1], 1.0)) # Assuming fixed std dev for simplicity

        # Generate binomial data
        data_binomial[i] = rand(Binomial(binomial_trials, true_p[k]))

        # Generate multinomial data
        data_multinomial[i, :] = rand(Multinomial(multinomial_total_count, true_θ[k]))
    end

    println("Synthetic Mixed Data Generated. First 5 observations:")
    display(hcat(data_continuous[1:5], data_binomial[1:5], data_multinomial[1:5, :], cluster_assignments_true[1:5]))

    return data_continuous, data_binomial, data_multinomial,
      num_clusters, num_cont_dims, binomial_trials, multinomial_total_count, multinomial_categories

  end

  if datatype=="timeseries_testdata"

    # Generate a Synthetic Time Series
    # Define parameters
    fs = 100        # Sampling frequency (samples per second)
    T = 2           # Duration of the signal (seconds)
    n = fs * T      # Number of samples
    x = range(0, T, length=n) # Time vector

    # Create a signal with multiple sine waves (harmonics) and some noise
    f1 = 5          # Frequency 1 (Hz)
    f2 = 12         # Frequency 2 (Hz)
    f3 = 20         # Frequency 3 (Hz)

    y = 1.0 * sin.(2π * f1 * x) .+ \
            0.5 * sin.(2π * f2 * x) .+ \
            0.2 * sin.(2π * f3 * x) .+ \
            0.1 * randn(n) # Add some random noise
    
    return x, y, fs

  end


  if datatype == "maunaloa_co2"

    # downloaded and saved from: https://scrippsco2.ucsd.edu/data/atmospheric_co2/primary_mlo_co2_record.html
    
    # timestamp of last save: May  1  2025
    fn = joinpath( project_directory, "data", "monthly_in_situ_co2_mlo.csv" ) 

    maunaloa = CSV.read(fn, DataFrame, skipto=65)

    maunaloa = maunaloa[!, [ 4,5]]
    rename!(maunaloa, [ :datestamp, :co2ppm])
    maunaloa.co2ppm = map(x -> x == -99.99 ? missing : x, maunaloa.co2ppm)
    maunaloa.co2 = (maunaloa.co2ppm .- mean(skipmissing(maunaloa.co2ppm))) ./ std(skipmissing(maunaloa.co2ppm))
    maunaloa.float_date = Float64.(maunaloa.datestamp)

    maunaloa = filter(:co2 => !ismissing, maunaloa)

    ### <<<<<<<<<<<<< IMPORTANT .. 
    ### presence of Missing Type confuses logpdf function, 
    ### this does an element-wise copy, droping the Missing Type (yes Types can be a pain)
    # x = map(v->v, maunaloa.float_date)
    # y = map(v->v, maunaloa.co2)  
    x = identity.(maunaloa.float_date)
    y = identity.(maunaloa.co2)
    ### <<<<<<<<<<<<< IMPORTANT .. 
    
    sampling_freq = 12 / 1 # 12 per year
    return x, y, sampling_freq

  end

  

  if datatype == "complex_testdata"

    # 1. Coordinates: Space (Xlon, Xlat) and Time (T)
    coords_space = rand(N, 2)
    coords_time = reshape(collect(1.0:N), :, 1)

    # 2. Covariates
    # Z: Purely spatial covariate
    Z = randn(N)

    # Latent (True) Spatiotemporal Covariates
    U1_true = sin.(coords_time[:,1] ./ 5.0) .+ 0.5 .* Z
    U2_true = cos.(coords_time[:,1] ./ 5.0) .- 0.3 .* Z
    U3_true = 0.2 .* (coords_time[:,1] ./ N) .+ 0.1 .* Z

    # 3. Add measurement error to covariates (observed version)
    sigma_u = 0.1
    U1_obs = U1_true .+ randn(N) .* sigma_u
    U2_obs = U2_true .+ randn(N) .* sigma_u
    U3_obs = U3_true .+ randn(N) .* sigma_u

    # 4. Generate Dependent Variable Y
    # Components: Linear Trend + Seasonal Harmonic + Latent Process + Noise
    trend = 0.05 .* coords_time[:,1]
    seasonal = 1.0 .* cos.(2 * pi .* coords_time[:,1] ./ period)

    # Simulate a spatial effect manually for the ground truth
    spatial_effect = sin.(coords_space[:,1] .* 2π) .* cos.(coords_space[:,2] .* 2π)

    sigma_y = 0.2
    # Y is a function of trend, season, GP effect, and U1
    y_obs = 1.0 .+ trend .+ seasonal .+ spatial_effect .+ (0.5 .* U1_true) .+ randn(N) .* sigma_y

    return (
        y_obs = y_obs,
        U1_obs = U1_obs,
        U2_obs = U2_obs,
        U3_obs = U3_obs,
        Z = Z,
        coords_space = coords_space,
        coords_time = coords_time
    )

  end

 
  if example_data=="multi-fidelity" 

    # 1. Define sample sizes for resolutions
    Ny, Nu, Nz = 40, 80, 120
    Nt_y, Nt_u = 4, 8
    Ns_y, Ns_u = 10, 10

    # 2. Mock coordinates for different resolutions
    coords_z = rand(Nz, 2)
    coords_u_s = rand(Ns_u, 2)
    coords_u_t = reshape(repeat(collect(1.0:Nt_u), inner=Ns_u), :, 1)
    coords_y_s = rand(Ns_y, 2)
    coords_y_t = reshape(repeat(collect(1.0:Nt_y), inner=Ns_y), :, 1)

    # 3. Mock observations
    y = randn(Ny)

    # Latent (True) Spatiotemporal Covariates

    z = randn(Nz)
    U1_true = sin.(coords_time[:,1] ./ 5.0) .+ 0.5 .* z
    U2_true = cos.(coords_time[:,1] ./ 5.0) .- 0.3 .* z
    U3_true = 0.2 .* (coords_time[:,1] ./ N) .+ 0.1 .* z

    # 3. Add measurement error to covariates (observed version)
    sigma_u = 0.1
    u1 = U1_true .+ randn(Nu) .* sigma_u
    u2 = U2_true .+ randn(Nu) .* sigma_u
    u3 = U3_true .+ randn(Nu) .* sigma_u


    return ( y, z, u1, u2, u3, 
      coords_z, coords_u_s, coords_u_t, coords_y_s, coords_y_t )

  end


end
 
