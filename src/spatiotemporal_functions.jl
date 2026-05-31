
 
function init_params_extract(X)
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    # print everything to console
    show(stdout, "text/plain", x) # display all estimates
end 
 

function firstindexin(a::AbstractArray, b::AbstractArray)
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
  
function β( mode, conc )
    # alternate parameterization of beta distribution 
    # conc = α + β     https://en.wikipedia.org/wiki/Beta_distribution
    beta1 = mode *( conc - 2  ) + 1.0
    beta2 = (1.0 - mode) * ( conc - 2  ) + 1.0
    Beta( beta1, beta2 ) 
end 
  
function modelruntime(o)
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
end
  
 

function turingindex( indices, sym=nothing, dims=nothing  ) 
     
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
end


function showtuples(X)
    for k in keys(X)
        val = getproperty(X, k)
        # Skip displaying keys with NaN values
        # Check if value is numeric before rounding to avoid errors
        display_val = val isa Number ? round(val, digits=3) : val
        println("$k: $display_val")
    end
end



function showparams(X, keywords=["rho", "phi", "sigma",  "mu_", "l_", "ls_"]; limit=10 )
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

# etas = [1 10 100 1000 1e+4 1e+5];
# d = size of matrix

# EXTENDED ONION METHOD to generate random correlation matrices
# distributed ~ det(S)^eta [or maybe det(S)^(eta-1), not sure]
# https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices

# LKJ modify this method slightly, in order to be able to sample correlation matrices C from a distribution proportional to [detC]η−1. The larger the η, the larger will be the determinant, meaning that generated correlation matrices will more and more approach the identity matrix. The value η=1 corresponds to uniform distribution. On the figure below the matrices are generated with η=1,10,100,1000,10000,100000. 

    beta = eta + (d-2)/2;
    u = rand( Beta(beta, beta) );
    r12 = 2*u - 1;
    S = [1 r12; r12 1];  

    for k = 3:d
        beta = beta - 1/2;
        y = rand( Beta((k-1)/2, beta) );  # sample from beta
        r = sqrt(y);
        theta = randn(k-1,1);
        theta = theta/norm(theta);
        w = r*theta;
        U, E = eigen(S);
        U = hcat(U)
        R = U' * sqrt(E) * U; # R is a square root of S
        q = R[].re * w;
        S = [S q; q' 1];
    end
    return S
end




function build_st_inputs(time_indices, space_indices, spatial_coords)
  # Space-Time Input Construction
  # Space and Time as continuous coordinates.
  # Inputs: 
  #   spatial_coords: Matrix (2 x N_nodes) -> [Lat, Lon]
  #   time_coords: Vector (T_steps)
  # Returns:
  #   ColVecs of 3D points (Time, Lat, Lon)

  # Map indices to actual coordinates
  # This assumes spatial_coords is 2xN

  # Extract coords for every observation
  coords = spatial_coords[:, space_indices] # 2 x y_N
  times = time_indices' # 1 x y_N

  # Stack to create 3D input: [Time; Lat; Lon]
  return ColVecs(vcat(times, coords))
end

 


function adjacency_matrix_to_nb( W )
    nau = size(W)[1]
    # W = LowerTriangular(W)  # using LinearAlgebra
    nb = [Int[] for _ in 1:nau]
    Threads.@threads for i in 1:nau
        nb[i] = findall( isone, W[i,:] )
    end
    return nb
end


function nb_to_adjacency_matrix( nb )
    nau = Integer( length( unique( reduce(vcat, nb) )) )
    W = zeros( Int8, nau, nau )
    Threads.@threads for i in 1:nau
        for j in 1:length( nb[i] )
            k = nb[i][j]
            W[i, k] = 1
        end
    end
    return(W)
end


function nodes( adj )
    nau = length(adj)
    N_edges = Integer( length( reduce(vcat, adj) )/2 )
    node1 =  fill(0, N_edges); 
    node2 =  fill(0, N_edges); 
    i_edge = 0;
    for i in 1:nau
        u = adj[i]
        num = length(u)
        for j in 1:num
            k = u[j]
            if i < k
                i_edge = i_edge + 1;
                node1[i_edge] = i;
                node2[i_edge] = k;
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = Graphs.adjacency_matrix(g)
    
    # D = diagm(vec( sum(W, dims=2) ))
    scalefactor = scaling_factor_bym2(W)

    return node1, node2, scalefactor
end




function scaling_factor_bym2( adjacency_mat )
    # re-scaling variance using Reibler's solution and 
    # Buerkner's implementation: https://codesti.com/issue/paul-buerkner/brms/1241)  
    # Compute the diagonal elements of the covariance matrix subject to the 
    # constraint that the entries of the ICAR sum to zero.
    # See the inla.qinv function help for further details.
    # Q_inv = inla.qinv(Q, constr=list(A = matrix(1,1,nbs$N),e=0))  # sum to zero constraint
    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    # scaling_factor = exp(mean(log(diag(Q_inv))))
    N = size(adjacency_mat)[1]
    asum = vec( sum(adjacency_mat, dims=2)) 
    asum = float(asum) + N .* max.(asum) .* sqrt(1e-15)  # small perturbation
    Q = Diagonal(asum) - adjacency_mat
    A = ones(N)   # constraint (sum to zero)
    S = Q \ Diagonal(A)  # == inv(Q)
    V = S * A
    S = S - V * inv(A' * V) * V'
    # equivalent form as inv is scalar
    # S = S - V / (A' * V) * V'
    scale_factor = exp(mean(log.(diag(S))))
    return scale_factor
 
end



function scaling_factor_bym2(node1, node2, groups=ones(length(node1))) 
    ## calculate the scale factor for each of k connected group of nodes, 
    ## copied from the scale_c function from M. Morris
    gr = unique( groups )
    n_groups = length(gr)
    scale_factor = ones(n_groups)
    Threads.@threads for j in 1:n_groups 
      k = findall( x -> x==j, groups)
      if length(k) > 1 
        e = Edge.(node1[k], node2[k])
        g = Graph(e)
        adjacency_mat = adjacency_matrix(g)
        scale_factor[j] = scaling_factor_bym2( adjacency_mat )
      end
    end
    return scale_factor
end
  

 

function sample_gaussian_process( ; GPmethod="cholesky", returntype="default",
    fkernal=nothing, kerneltype="default", kvar=nothing, kscale=nothing, gpc=GPC(),
    Yobs, Xobs, Xinducing=nothing, lambda=0.0001 )
    
    if isnothing(fkernal)
        if kerneltype=="default" || kerneltype=="squared_exponential"
            fkernal = kvar * SqExponentialKernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
        if kerneltype=="matern32"
            fkernal = kvar * Matern32Kernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
    end


    if GPmethod=="textbook"
        # mean process at predictons Xobs
        Ko = kernelmatrix( fkernal, vec(Xobs) ) 
        Kcommon = inv(Ko + lambda*I)   # Note already inversed taken

        if !isnothing(Xinducing)

            Ki = kernelmatrix( fkernal, vec(Xinducing) )   
            Kio = kernelmatrix( fkernal, vec(Xinducing), vec(Xobs) )   # transfer to inducing points
            Yinducing_mean_process = Kio * Kcommon * Yobs   # mean process at inducing points
            # covariance at predictions Covp:
            # Covp = Ki - Kio * inv(Ko + lambda*I ) * Kio' 
            Covi = Symmetric( Ki - Kio * Kcommon * Kio'  + lambda*I ) # note Ccommon is already inverted 
            MVNi = MvNormal( Yinducing_mean_process, Covi )

            Yinducing_sample  = rand( MVNi )
            Li =  cholesky(Symmetric( Ki + lambda*I)).L   # cholesky on inducing locations  

            Yobs_mean_process =  Kio' * ( Li' \ (Li \ Yinducing_mean_process  ) )  # back to original locations
            Covo = Symmetric(cov(kernelmatrix( fkernal,  vec(Xobs) )) + lambda*I)
            MVN = MvNormal(Yobs_mean_process, Covo)  # of observations

            if returntype=="fcovariance"  
                return MVN
            end

            Yobs_sample =  Kio' * ( Li' \ (Li \ Yinducing_sample  ) )  # back to original locations

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return (MVN=MVN, MVNi=MVNi, Li=Li, loglik=LogLik,
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else
             
            mean_process = Ko * Kcommon * Yobs   # mean process            
            MVN = MvNormal(mean_process, Ko + lambda*I  ) # lambda*I creates a diagonal matrix

            if returntype=="fcovariance"  
                return MVN  # of observations
            end

            Yobs_sample = rand( MVN ) # sample
            
            if returntype=="sample"
                return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod )
            end
            
            LogLik = logpdf(MVN,Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return ( MVN=MVN, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)

        end 
 
    end

    if GPmethod=="cholesky"
        # this avoids inversion of the big covariance and re-uses cholesky factors 
        if !isnothing(Xinducing)
            Ko = kernelmatrix( fkernal, vec(Xobs) ) 

            Ki = kernelmatrix( fkernal, vec(Xinducing) ) 
            Kio = kernelmatrix( fkernal, vec(Xinducing), vec(Xobs) ) # transfer to inducing points
            Lo = cholesky(Symmetric( Ko + lambda*I)).L 
            Li = cholesky(Symmetric( Ki + lambda*I)).L   # cholesky on inducing locations  
            Yinducing_mean_process  = Kio * ( Lo' \ (Lo \ Yobs ) )  # == mean_process mean latent process

            Covi = Symmetric( cov(Ki) + lambda*I)  
            MVN = MvNormal( Yinducing_mean_process, Covi )

            if returntype=="fcovariance" 
                return MVN
            end

            Yobs_mean_process = Kio' * ( Li' \ (Li \ Yinducing_mean_process ))  # mean process from inducing s_coord_tuple
            Yinducing_sample  = Yinducing_mean_process + Li * rand(Normal(0, 1), size(Li,1))   # faster sampling without covariance
            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2)) # error process 

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yinducing_mean_process)
            
            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end
            
            return (MVN=MVN, Li=Li, Lo=Lo, loglik=LogLik,
                    Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else

            Ko = kernelmatrix( fkernal, vec(Xobs) ) 
            Lo = cholesky(Symmetric( Ko + lambda*I)).L 
            Yobs_mean_process = Ko' * ( Lo' \ (Lo \ Yobs ))  # mean process from inducing s_coord_tuple
            
            Covo = Symmetric( cov(Ko) + lambda*I)  
            MVN = MvNormal( Yobs_mean_process, Covo ) # of observations

            if returntype=="fcovariance" 
                return MVN
            end

            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2)) # error process 

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)
       
            if returntype=="sample_loglik"
                return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
            end

            return (MVN=MVN, Lo=Lo, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)

        end
    end
 

    if GPmethod=="GPexact"

        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp(Xobs, lambda)

        if returntype=="fcovariance"
            return fobs
        end 

        fposterior = posterior(fobs, Yobs) 
        
        if returntype=="posterior"
            return fposterior
        end

        Yobs_sample =  rand(fposterior(Xobs, lambda) )   

        if returntype=="sample"
            return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fobs, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, fposterior=fposterior, Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
    end
 
    if GPmethod=="GPsparse"
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        fsparse = SparseFiniteGP(fobs, finducing)

        if returntype=="fcovariance"
            return fsparse
        end 

        fposterior = posterior(fsparse, Yobs)

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fsparse, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

    end

    if GPmethod=="GPvfe" # Variational Free Energy
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp(Xinducing, lambda )
        fsparse = VFE( finducing )

        if returntype=="fcovariance"
            return fsparse
        end 
        
        fposterior = posterior(fsparse, fobs, Yobs)  # Distribution is MvNormal  

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end
        
        LogLik = AbstractGPs.elbo(fsparse, fobs, Yobs)  # to a constant
      
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
    end
      
end
 
 


function turing_glm_icar_summary( method="mcmc"; 
    GPmethod="cholesky", family="poisson", 
    Y=nothing, YG=nothing,  
    msol=nothing, model=nothing,
    X=nothing, G=nothing, Gp=nothing, nInducing=nothing, log_offset=nothing, good=nothing,
    scaling_factor=nothing, n_sample=nothing, nAU=nothing, auid=nothing, tuid=nothing, 
    kerneltype="squared_exponential"
)
 
    fixed_effects = nothing
    sp_re_structured = nothing
    sp_re_unstructured = nothing
    Gymu = nothing

    # Main.DEBUG = Ymu
  
    if !isnothing(good)
                     
        if !isnothing(Y)
            Y = Y[good] 
        end
        
        if !isnothing(YG)
            YG = YG[good] 
        end
        if !isnothing(X)
            X = X[good,:] 
        end
        if !isnothing(G)
            G = G[good,:] 
        end
        if !isnothing(log_offset)
            log_offset = log_offset[good] 
        end
        if !isnothing(auid)
            auid = auid[good] 
        end
        if !isnothing(tuid)
            tuid = tuid[good] 
        end

    end

    if !isnothing(X)
        nX = size(X,2)
        nData = size(X,1)
        fixed_effects = zeros(nX, n_sample)
    end

    if method=="mcmc"
        nchains = size(msol)[3]
        nsims = size(msol)[1]
    end

    if method=="variational_inference"
        res = rand(msol, n_sample)
        nsims = size(res)[2]
    end

    if method=="optim"
        res = hcat( vec(msol.values) )
        nsims = size(res)[2]
    end

    if isnothing(n_sample)
        n_sample = nsims         # do all
    end

    if !isnothing(G)
        nG = size(G)[2]
        if isnothing(X) # in case no fixed effects
            nData = size(G,1)
        end
        if isnothing(nInducing)
            nInducing = size(G,2)
        end
        Gymu =zeros(nInducing, nG, n_sample)
    end

    if !isnothing(auid)
        if isnothing(nAU)
            nAU = length(auid)
        end
        sp_re_structured = zeros(nAU, n_sample) 
        sp_re_unstructured = zeros(nAU, n_sample) 
    end


    ntries_mult=2
    ntries = 0
    z = 0

    Ypred = zeros(nData, n_sample) 
    
    if method=="mcmc"

        while z <= n_sample 
            ntries += 1
            ntries > ntries_mult * n_sample && break 
            z >= n_sample && break

            j = rand(1:nsims)  # nsims
            l = rand(1:nchains) # nchains

            Ymu = zeros( nData ) 

            if !isnothing(X)
                # fixed effects
                # beta = Array(msol[:, turingindex( model, :beta), :] )
                beta  = [ msol[j, Symbol("beta[$k]"), l]  for k in 1:nX]
                Ymu += X * beta
                # Main.DEBUG = beta
            end

            if !isnothing(auid)
                theta = [ msol[j, Symbol("theta[$k]"), l] for k in 1:nAU]
                phi   = [ msol[j, Symbol("phi[$k]"), l]   for k in 1:nAU]
                sigma = msol[j, Symbol("sigma"), l] 
                rho   = msol[j, Symbol("rho"), l] 
                sp_re_besag = sigma .* (sqrt.(rho ./ scaling_factor) .* phi )  
                sp_re_iid   = sigma .* (sqrt.(1 .- rho) .* theta )
                sp_re = sp_re_besag + sp_re_iid  
                Ymu += sp_re[auid] 
            end
            
            if !isnothing(log_offset)
                Ymu .+= log_offset 
            end

            if !isnothing(G)

                # gaussian process for covariates G
                kernel_var = [ msol[j, Symbol("kernel_var[$k]"), l]  for k in 1:nG] 
                kernel_scale = [ msol[j, Symbol("kernel_scale[$k]"), l]  for k in 1:nG] 
                
                if any( occursin.("l2reg", String.(names(msol))) )
                    l2reg = [ msol[j, Symbol("l2reg[$k]"), l]  for k in 1:nG]  
                else
                    l2reg = fill(1.0e-4, nG)                    
                end
                
                Gymu_s = zeros( nInducing, nG)
                YGcurr = YG - Ymu
                for i in 1:nG
                    # Kfn = fkernal( kernfunctype, (kernel_var[i], kernel_scale[i]) ) 
                    ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample",
                        kerneltype=kerneltype, kvar=kernel_var[i], kscale=kernel_scale[i],
                        Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i], 
                    )
                    
                    # Main.DEBUG = ys
                    Ymu += ys.Yobs_sample
                    Gymu_s[:,i] = ys.Yinducing_sample 
                end
            end
  
            z += 1

            if family=="poisson"
                ineg = findall(x->x<0, Ymu)
                if length(ineg)>0
                    Ymu[ineg] .= 0.0
                end
                Ypred[:,z] = rand.(LogPoisson.(Ymu));   
            elseif family=="bernoulli"
                Ypred[:,z] = rand.(Bernoulli.( logistic.(Ymu) ) ) 
            elseif family=="gaussian"
                Ysd = msol[j, Symbol("Ysd"), l] 
                Ypred[:,z] = rand.(Normal.( Ymu, Ysd ) ) 
            end

            if !isnothing(X)
                fixed_effects[:,z] = beta
            end

            if !isnothing(auid)
                if !isnothing(sp_re_structured)
                    sp_re_structured[:,z]   = sp_re_besag
                    sp_re_unstructured[:,z] = sp_re_iid
                end
            end

            if !isnothing(G)
                Gymu[:,:,z] = Gymu_s
            end

        end  # while
    
        if z < n_sample 
            @warn  "Insufficient number of solutions" 
        end
        res = Array(msol)
    end


    if method=="variational_inference"
        # variational inference method

        # this in case some samples provide failed predictions (e.g., not PD, etc)
        while z <= n_sample 
            ntries += 1
            ntries > ntries_mult * n_sample && break 
            z >= n_sample && break

            l = rand(1:nsims) # nchains

            Ymu = zeros( nData ) 

            if !isnothing(X)
                # fixed effects
                beta  = [ msol[j, Symbol("beta[$k]"), l]  for k in 1:nX]
                beta = res[ turingindex( model, :beta ), l ]
                Ymu += X * beta
            end

            if !isnothing(auid)
                theta = res[ turingindex( model, :theta ), l ]
                phi   = res[ turingindex( model, :phi ), l ]
                sigma = res[ turingindex( model, :sigma ), l ] 
                rho   = res[ turingindex( model, :rho ), l ] 
                sp_re_besag = sigma .* (sqrt.(rho ./ scaling_factor) .* phi )  
                sp_re_iid   = sigma .* ( sqrt.(1 .- rho) .* theta )
                sp_re = sp_re_besag + sp_re_iid  
                Ymu += sp_re[auid] 
            end
            
            if !isnothing(log_offset)
                Ymu .+= log_offset 
            end

            if !isnothing(G)
                # gaussian process for covariates G

                kernel_var = res[ turingindex( model, :kernel_var )] 
                kernel_scale = res[ turingindex( model, :kernel_scale )]
                if any( occursin.("l2reg", String.(names(msol))) )
                    l2reg = res[ turingindex( model, :l2reg )]
                else
                    l2reg = fill(1.0e-4, nG)                
                end
                Gymu_s = zeros( nInducing, nG)
                YGcurr = YG - Ymu
                for i in 1:nG
                    # Kfn = fkernal( kernfunctype, (kernel_var[i], kernel_scale[i]) ) 
                    ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample",
                        kerneltype=kerneltype, kvar=kernel_var[i], kscale=kernel_scale[i],
                        Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i], 
                    ) 
                    # Main.DEBUG = ys
                    Ymu += ys.Yobs_sample
                    Gymu_s[:,i] = ys.Yinducing_sample 
                end
            end
        
    
            z += 1

            if family=="poisson"
                Ypred[:,z] = rand.(LogPoisson.(Ymu));   
            elseif family=="bernoulli"
                Ypred[:,z] = rand.(Bernoulli.( logistic.(Ymu) ) ) 
            elseif family=="gaussian"
                Ysd = res[j, Symbol("Ysd"), l] 
                Ypred[:,z] = rand.(Normal.( Ymu, Ysd ) ) 
            end
    
            if !isnothing(X)
                fixed_effects[:,z] = beta
            end

            if !isnothing(auid)
                sp_re_structured[:,z]   = sp_re_besag
                sp_re_unstructured[:,z] = sp_re_iid
            end

            if !isnothing(G)
                Gymu[:,:,z] = Gymu_s
            end
            
        end  # while
    
        if z < n_sample 
            @warn  "Insufficient number of solutions" 
        end

    end

    if method =="optim"
        # optim method 

        # this in case some samples provide failed predictions (e.g., not PD, etc)
        while z <= n_sample 
            ntries += 1
            ntries > ntries_mult * n_sample && break 
            z >= n_sample && break

            l = rand(1:nsims) # nchains

            Ymu = zeros( nData ) 

            if !isnothing(X)
                # fixed effects
                beta = res[ turingindex( model, :beta ), l ]
                Ymu += X * beta
            end

            if !isnothing(auid)
                theta = res[ turingindex( model, :theta ), l ]
                phi   = res[ turingindex( model, :phi ), l ]
                sigma = res[ turingindex( model, :sigma ), l ] 
                rho   = res[ turingindex( model, :rho ), l ] 
                sp_re_besag = sigma .* (sqrt.(rho ./ scaling_factor) .* phi )  
                sp_re_iid   = sigma .* ( sqrt.(1 .- rho) .* theta )
                sp_re = sp_re_besag + sp_re_iid  
                Ymu += sp_re[auid] 
            end
            
            if !isnothing(log_offset)
                Ymu .+= log_offset 
            end

            if !isnothing(G)
                # gaussian process for covariates G

                kernel_var = res[ turingindex( model, :kernel_var )] 
                kernel_scale = res[ turingindex( model, :kernel_scale )]

                if any( occursin.("l2reg", String.(names(msol.values)[1] )) )
                    l2reg = res[ turingindex( model, :l2reg )]
                else
                    l2reg = fill(1.0e-4, nG)             
                end
                Gymu_s = zeros( nInducing, nG)
                YGcurr = YG - Ymu
                for i in 1:nG
                    # Kfn = fkernal( kernfunctype, (kernel_var[i], kernel_scale[i]) ) 
                    ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample",
                        kerneltype=kerneltype, kvar=kernel_var[i], kscale=kernel_scale[i],
                        Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i], 
                    )
                    # Main.DEBUG = ys
                    Ymu += ys.Yobs_sample
                    Gymu_s[:,i] = ys.Yinducing_sample 
                end

            end    
            
            z += 1

            if family=="poisson"
                Ypred[:,z] = rand.(LogPoisson.(Ymu));   
            elseif family=="bernoulli"
                Ypred[:,z] = rand.(Bernoulli.( logistic.(Ymu) ) ) 
            elseif family=="gaussian"
                Ysd = res[j, Symbol("Ysd"), l] 
                Ypred[:,z] = rand.(Normal.( Ymu, Ysd ) ) 
            end
    
            if !isnothing(X)
                fixed_effects[:,z] = beta
            end

            if !isnothing(auid)
                sp_re_structured[:,z]   = sp_re_besag
                sp_re_unstructured[:,z] = sp_re_iid
            end

            if !isnothing(G)
                Gymu[:,:,z] = Gymu_s
            end
            
        end  # while
    
        if z < n_sample 
            @warn  "Insufficient number of solutions" 
        end
    end

    return ( 
        Ypred=Ypred, 
        fixed_effects=fixed_effects,
        sp_re_unstructured=sp_re_unstructured, 
        sp_re_structured=sp_re_structured, 
        res=res, 
        Gymu=Gymu
    )

end

 

function plot_variational_marginals(z, sym2range)
    # copied straight from https://turinglang.org/docs/tutorials/variational-inference/
    ps = []

    for (i, sym) in enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            k = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$k]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density",
                    margin=1.5mm,
                )
                push!(ps, p)
                k += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density",
                margin=1.5mm,
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 2000), margin=4.0mm)
end


function fkernal( kernfunctype="squared_exp", params=nothing )

    if kernfunctype=="squared_exp"
        out = params[1] * SqExponentialKernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern12"
        out = params[1] * Matern12Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern32"
        out = params[1] * Matern32Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern52"
        out = params[1] * Matern52Kernel() ∘ ScaleTransform(params[2])  
    end

    # ∘ ARDTransform(α)

    return out
end


sekernel2(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

sekernel(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);
 

# Squared-exponential covariance function
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
exp_cov_fn(D, phi) = exp.(-D / phi)



# generic kernel functions 

# lenscale = -1 / log(ρ)
# σ_ar1^2 / (1 - ρ^2) = marginal variance
# kernel_ar1(σ, ρ) = σ^2 * with_lengthscale(Matern12Kernel(), -1/log(ρ)) 
# the softplus should not be necessary ... 

kernel_ar1(σ, ρ) = σ^2 / softplus(1 - ρ^2) * with_lengthscale(Matern12Kernel(), softplus(-1 / log(ρ)) )

# RW2 is equivalent to a Spline kernel or an Integrated Wiener Process
# For simplicity, we often use a high-order Matern or a custom structure

kernel_rw2(σ) = σ^2 * Matern52Kernel() # Matern32 is a common smooth approximation for RW2


   
    
function assign_time_units(t_coord; method="regular", t_N=nothing, u_N=12)

    if method=="regular"

        tint = Int.(floor.(t_coord))
        t0, t1 = minimum(tint), maximum(tint)
        t_n = t1-t0
        if !isnothing(t_N) 
            if t_n != t_N
                print("warning: time range and unique years do not match")
            end
        end

        t_idx = tint .- t0 .+ 1
        t_vals = collect(t0:t1) .- t0 .+ 1
        t_yr = collect(t0:t1)
        t_brks = (t_yr, t1+1)
        t_mids = t_yr .+ 0.5
        
        u_coord = t_coord - tint

        u_disc = discretize_data( u_coord, N_cat=u_N, method="regular" )  # seasonality discretized

        return (
            t_coord = t_coord, 
            t_idx = t_idx, 
            t0=t0, 
            t1=t1, 
            t_vals, 
            t_yr=t_yr, 
            t_mids=t_mids, 
            t_brks=t_brks,
            tn=length(t_vals), 
            u_coord=u_coord, 
            u_idx=u_disc.idx, 
            u_brks=u_disc.brks,
            u_mids=u_disc.mids, 
            u_vals=collect(1:u_N) 
        )
    end

end


function discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv = 0, maxv=1)
     
    if method=="quantile" 
        # simpler solutions
        probs = isnothing(probs) ? collect(range(0.0, stop=1.0, length=N_cat + 1)) : probs
        brks = isnothing(brks) ? quantile(X, probs) : brks
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs )

    elseif method == "regular"
        dx = isnothing(dx) ? (maxv - minv) / N_cat : dx
        probs = nothing
        brks = collect(minv:dx:maxv)
        mids = brks[1:N_cat] + diff(brks) 
        idx = map(x -> clamp(searchsortedfirst(brks, x) - 1, 1, N_cat), X)
        return ( idx=idx, brks=brks, mids=mids, probs, dx=dx )
    
    elseif method=="regular_resolution"    

        xd = round.(Int, X ./ dx ) .* dx   # resolution to  units of dx
        brks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
        mids = midpoints(brks)
        N_cats = length(mids)
        
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        return xd, xi, mids, N_cats, dx

    elseif method=="quantile_resolution"
    
        brks = quantile(X, range(0, 1, length=N_cats+1))
        mids = midpoints(brks)
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        dx = diff(mids)[1]
        xd = mids[xi] 
        return xd, xi, mids, N_cats, dx

    end


end
 
 

function assign_covariate_levels( X; N_cat=9, brks=nothing, probs=nothing )
    X = Array(X)
    U = [ discretize_data( X[:,i], N_cat=N_cat, brks=brks, probs=probs ) for i in 1:size(X, 2) ]
    V = hcat([t[:idx] for t in  U]...)  # as matrix
    B = hcat([t[:brks] for t in  U]...)
    M = hcat([t[:mids] for t in  U]...)
    P = hcat([t[:probs] for t in  U]...)
    return( idx=V, brks=B, mids=M, probs=P )
end



function estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - s_coord_tuple: Vector of (x, y) coordinates for all time points.
    - t_idx: Vector of time indices corresponding to s_coord_tuple.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice
    filtered_pts = [p for (i, p) in enumerate(s_coord_tuple) if t_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for the target time slice $target_ts")
    end
    xs, ys = [p[1] for p in filtered_pts], [p[2] for p in filtered_pts]
    # Calculate bandwidth based on standard deviation of points
    bw_x = std(xs) * sd_extension_factor
    bw_y = std(ys) * sd_extension_factor
    # Define grid boundaries extending slightly beyond the data range
    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y
    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)
    # Gaussian KDE implementation
    for i in 1:grid_res
        for j in 1:grid_res
            x_val, y_val = x_grid[i], y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end
    # Normalize intensity to sum to 1 (optional, depending on desired output)
    intensity ./= sum(intensity)
    return x_grid, y_grid, intensity
end

function calculate_metrics(au)
    # Restoration: Calculate assignments and counts based on the actual centroids in the au object
    assigns = [argmin([sum((p .- c).^2) for c in au.centroids]) for p in au.s_coord_tuple]
    counts = [count(==(i), assigns) for i in 1:length(au.centroids)]

    # Safety: Filter valid counts to prevent downstream NaN propagation
    valid_counts = filter(x -> !isnan(x) && !ismissing(x), counts)

    if isempty(valid_counts)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end

    m_dens = mean(valid_counts)
    s_dens = std(valid_counts)
    cv_dens = s_dens / (m_dens + 1e-9)

    return (mean_density=m_dens, sd_density=s_dens, cv_density=cv_dens)
end


function get_spatial_graph( centroids, adjacency_edges )
    """
    Synopsis: Converts partitioning results into a formal SimpleGraph. 
    Outputs: A SimpleGraph object.
    """
    n = length(centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(centroids))
    for edge in adjacency_edges
        xi, yi = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if xi > 0 && yi > 0 add_edge!(g, xi, yi) end
    end
    return g
end



function plot_kde_simple(s_coord_tuple; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - s_coord_tuple: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy t_idx of 1s since we are plotting a static slice
    t_idx_dummy = ones(Int, length(s_coord_tuple))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                  title=title,
                  c=:viridis,
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end




function scottish_lip_cancer_data_spacetime(n_years::Int=10; rndseed::Int=42)
    # "expand" scottish lip cancer data to a space-time version
    # original data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    Random.seed!(rndseed)

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
    # D = diagm(vec(sum(W, dims=2)))
 
    au = assign_spatial_units( W ) # "infer" from the adjacency network (W)
    pts_base = au.centroids
    
    N_total = nAU * n_years

    # 1. Random Walk Trend
    rw_trend = cumsum(randn(n_years) .* 0.5)

    # 2. Expand Data Vectors
    y_expanded = repeat(y_base, n_years)
    E_expanded = repeat(E_base, n_years)
    x_expanded = repeat(x_base, n_years)
    t_idx = repeat(1:n_years, inner=nAU)
    s_coord_tuple = repeat(pts_base, n_years)
    # The s_idx is the spatial unit identifier (1 to 56)
    s_idx = repeat(1:nAU, n_years)
 
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
        y=y_final, X=X, log_offset=log_offset, t_idx=t_idx,
        s_idx=s_idx, n_years=n_years, s_coord_tuple=s_coord_tuple, W=W, au=au
    )
end
  

function get_initial_parameters(model::DynamicPPL.Model; n_samples=100)
    # 1. Take multiple prior samples to find a robust center. 
    # This avoids starting in extreme tails for diffuse priors.
    samples = [Dict(pairs(rand(model))) for _ in 1:n_samples]
    
    init_dict = Dict{Symbol, Any}()
    for k in keys(samples[1])
        ks = Symbol(k) # Ensure the key is coerced to a Symbol for the dictionary
        vals = [s[k] for s in samples]
        
        # Detect if it's a fixed parameter (Dirac) - variance is zero
        if all(v -> v == vals[1], vals)
            init_dict[ks] = vals[1]
            continue
        end

        mu = mean(vals)
        s_name = string(ks)

        # 2. Apply expert heuristics to refine stochastic parameters
        if vals[1] isa AbstractVector
            # Latent fields: Start at exactly zero (the mathematical mean)
            # This prevents initial energy spikes in gradient-based samplers.
            init_dict[ks] = zeros(eltype(mu), length(mu))
        elseif occursin(r"sigma|ls_|lengthscale", s_name)
            # Scales: Ensure a healthy positive start
            init_dict[ks] = max(0.1, mu) 
        elseif occursin("rho", s_name)
            # Correlations: Clamp to avoid boundary issues (e.g., rho=1.0)
            init_dict[ks] = clamp(mu, -0.9, 0.9)
        elseif occursin("phi_zi", s_name)
            # Zero-inflation: Typically low
            init_dict[ks] = clamp(mu, 0.01, 0.2)
        else
            init_dict[ks] = mu
        end
    end
    return init_dict
end




function parameter_inits(model::DynamicPPL.Model; n_samples=10, 
    optimizer=SimulatedAnnealing(), #  
    maxiters = 500,     # Max optimization steps
    maxtime  = 60.0,    # Max allowed time in seconds
    show_trace = false,   # Print progress to the console
    kwargs...)

"""
    parameter_inits(model; n_samples=10, optimizer=LBFGS())

Refines heuristic initial parameters using Maximum A Posteriori (MAP) estimation.
Starts from the robust center found by `get_initial_parameters` and moves 
toward the peak of the posterior density. 

This is highly recommended for complex spatiotemporal models to prevent 
initial energy spikes that can crash the NUTS sampler.
"""

    # 1. Get the robust heuristic starting point
    init_guess = get_initial_parameters(model; n_samples=n_samples)
    
    # 2. Use MAP to find the local mode (peak density)
    try
        println("Trying MAP to improve starting point...")
        # Set link=false to prevent Dirac bijector crashes and fix initial_params keyword
        map_res = maximum_a_posteriori(model, optimizer; link=true, 
            initial_params=DynamicPPL.InitFromParams(NamedTuple(init_guess)), 
            show_trace=show_trace, iterations=maxiters, maxtime=maxtime, kwargs...) 

        return DynamicPPL.InitFromParams(NamedTuple(map_res.params))
    catch e
        @warn "MAP refinement failed: $e. Falling back to heuristic guess."
        return DynamicPPL.InitFromParams(NamedTuple(init_guess))
    end

end




function icar_form(theta, phi, sigma, rho)
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    # Reibler parameterization: https://pubmed.ncbi.nlm.nih.gov/27566770/
    # https://www.jstatsoft.org/index.php/jss/article/view/v063c01/841
    sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )  
end
   


 
function kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s, sigma_s, ls_t, sigma_t, noise_vec)
    # Spatial Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-4*I)
    Q_s = inv(K_s)

    # Temporal Precision
    k_t = Matern32Kernel() ∘ ScaleTransform(inv(ls_t))
    K_t = Symmetric(sigma_t^2 * kernelmatrix(k_t, unique_t) + 1e-4*I)
    Q_t = inv(K_t)

    # Full Kronecker Precision (Dense for AD compatibility)
    Q_full = Symmetric(Matrix(kron(Q_t, Q_s)) + 1e-4*I)
    L_q = cholesky(Q_full)

    # Sample: f = (L')^-1 * noise
    return L_q.U \ noise_vec
end


function get_posterior_means(ch, param_base, N)
    # Description:
    #   Extracts and averages posterior samples for a specific vector parameter.
    # Inputs:
    #   ch: MCMC sample chain.
    #   param_base: String prefix of the parameter (e.g., "s_eff").
    #   N: Length of the vector parameter.
    # Outputs:
    #   Vector of posterior means.

    means = zeros(N)
    
    for i in 1:N
        p_symbol = Symbol("$param_base[$i]")
        if p_symbol in names(ch, :parameters)
            means[i] = mean(ch[p_symbol])
        else
            @warn "Parameter $p_symbol not found in chain."
        end
    end
    
    return means
end



function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")

    # Helper function to generate inducing points (simple random sampling for now)

    Random.seed!(seed)
    n_data = size(coords, 1)
    if M_inducing >= n_data
        return coords # If M >= N, just use all data points (becomes exact GP)
    end

    
    if method=="random"
        indices = sample(1:n_data, M_inducing, replace=false)
        return coords[indices, :]
    end

    if method=="kmeans"
        #   Identifies optimal inducing point locations using K-Means clustering.
        #   Essential for Sparse/FITC Gaussian Process models.
        # Inputs:
        #   coords: N x D matrix of spatiotemporal coordinates.
        #   M_inducing: Target number of inducing points.
        # Outputs:
        #   M x D matrix of cluster centroids.

        # Transpose for Clustering.jl compatibility
        data_matrix = Matrix(coords')
        
        # Execute K-Means
        clustering_result = kmeans(data_matrix, M_inducing, maxiter=200)
        
        # Extract centroids and transpose back
        inducing_points = clustering_result.centers'
        
        return inducing_points
    end

    if method=="furthest_point"

        # Initialize with a random point
        inducing_points_idx = [rand(1:n_data)]
        distances = fill(Inf, n_data)

        # Convert coords to the expected format for pairwise
        coords_matrix = permutedims(coords)

        for _ in 2:M_inducing
            # Calculate distances from all points to the newest inducing point
            last_added_idx = inducing_points_idx[end]
            new_distances = colwise(Euclidean(), coords_matrix, coords_matrix[:, last_added_idx])

            # Update minimum distances to any inducing point found so far
            distances = min.(distances, new_distances)

            # Find the point farthest from any existing inducing point
            farthest_idx = argmax(distances)
            push!(inducing_points_idx, farthest_idx)
        end

        return coords[inducing_points_idx, :], inducing_points_idx

    end

end

 
function ar1_covariance(n, rho, var, ::Type{T}=Float64) where {T}
    # Description:
    #   Generates a full AR1 covariance matrix.
    # Inputs:
    #   n: Number of time points.
    #   rho: Correlation coefficient.
    #   var: Marginal variance.
    # Outputs:
    #   n x n Covariance matrix.

    vcv = zeros(T, n, n) .+ I(n)
    
    Threads.@threads for r in 1:n
        for c in 1:n
            if r >= c
                vcv[r, c] = var * rho^(r - c)
            end
        end
    end
    
    return Symmetric(vcv)
end

 


function ar1_covariance_local( n, rho, var,  ::Type{T}=Float64 )  where {T} 
    vcv = zeros( T, n, n) .+ I(n) 
    Threads.@threads for r in 1:n
    for c in 1:n
        d = r-c
        if d == 0 | d == 1
            vcv[r,c] = var * rho^d  
        end
    end
    end
    return vcv
end



function gp_predictions(; Y, D, mu, sig2, phi, cov_fn=exp_cov_fn, nN=length(Xnew), nP=size(res, 1) ) 
    ynew = Vector{Float64}()
    # Threads.@threads -- to add 
    for i in sample(1:size(res,1), nP, replace=true)
        K = cov_fn(D, phi[i])
        Koo_inv = inv(K[(nN+1):end, (nN+1):end])
        Knn = K[1:nN, 1:nN]
        Kno = K[1:nN, (nN+1):end]
        C = Kno * Koo_inv
        mvn = MvNormal( 
            C * (Y .- mu[i]) .+ mu[i], 
            Matrix(LinearAlgebra.Symmetric(Knn - C * Kno')) + sig2[i] * LinearAlgebra.I 
        ) 
        ynew = vcat(ynew, [rand(mvn) ] )
    end
    ynew = stack(ynew, dims=1)  # rehape to matrix   
    return ynew
end




function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )

    # Fit via ADVI. minor speed benefit vs NUTS
    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

    # vectorize variable names ... needs more conditions if 2-D or higher ..
    vns = []
    for (i, sym) in enumerate(vars) 
        j = union(indices[sym]...)  # <= array of ranges
        nj = sum(length.(j)) 
        if  nj > 1
            k = 1
            for r in j
                push!(vns, "$(sym)[$k]")
                k += 1
            end
        else
            push!(vns, "$(sym)") 
        end
    end
    
    vns = Symbol.(vns)

    msamples = rename(msamples, vns)

    mmean = combine( msamples, [ n => (x -> mean(x)) => n for n in names(msamples)  ] )
    mstd  = combine( msamples, [ n => (x -> std(x)) => n for n in names(msamples)  ] )

    out = (
        msol = msol,
        msamples = msamples, 
        mmean = mmean,
        mstd = mstd
    )
    
    return out
 
end

  


function rff_map(coords, W, b)
    # Description:
    #   Maps input coordinates into a Random Fourier Feature (RFF) space
    #   to approximate a kernel function (usually Squared Exponential).
    # Inputs:
    #   coords: N x D matrix of input features (space/time).
    #   W: D x M weight matrix sampled from spectral density.
    #   b: Vector of M random phases.
    # Outputs:
    #   N x M feature matrix.

    # Project coordinates into higher dimensional space
    projection = (coords * W) .+ b'
    
    # Apply cosine transformation with scaling factor
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    
    return feature_map
end

function generate_informed_rff_params(coords, M_rff_count)
    D_in = size(coords, 2)
    std_coords = vec(std(coords, dims=1)) .+ 1e-6
    W_fixed = randn(D_in, M_rff_count) ./ std_coords
    b_fixed = rand(M_rff_count) .* 2pi
    return W_fixed, b_fixed
end

function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 

macro save_carstm_state(file_to_save_name_sym)
  quote
    try
      # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
      local filename_val = $(esc(file_to_save_name_sym))
      @info "Saving CARSTM state to $(filename_val)..."
      # JLD2.@save expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@save "$(filename_val)" areal_units mod chain s_coord_tuple y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
      @info "CARSTM state saved successfully."
    catch e
      @error "Error saving CARSTM state: $e"
    end
  end
end

macro load_carstm_state(filename_sym)
  quote
    # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
    local filename_val = $(esc(filename_sym))
    if !isfile(filename_val)
      @error "File $(filename_val) not found."
      return nothing
    end
    try
      @info "Loading CARSTM state from $(filename_val)..."
      # JLD2.@load expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@load "$(filename_val)" areal_units mod chain s_coord_tuple y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
      @info "CARSTM state loaded successfully."
      # Variables are loaded directly into the calling scope by JLD2.@load
      # No explicit return value from the macro itself, as it injects variables
    catch e
      @error "Error loading CARSTM state: $e"
      return nothing
    end
  end
end


function init_params_extract( res=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # Description: Extracts initial parameter values from a model result summary or loads them from a file.
  # Inputs:
  #   - res: Model result object (default: NaN).
  #   - load_from_file: Boolean, if true loads params from fn_inits.
  #   - override_means: Boolean, if true applies custom overrides for specific parameter patterns.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the extracted or loaded mean parameter values.

  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns ); vns[u]
    if length(u) > 0  
      means[u] = sigma_prior  # from basic pca
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = v_prior  # from basic pca
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_copy( res=NaN, res0=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # using spatial parts of res0 
  # Description: Copies parameter values from a reference result (res0) to a target result structure (res).
  # Inputs:
  #   - res: Target model result object.
  #   - res0: Reference model result object.
  #   - load_from_file: Boolean to load from fn_inits instead.
  #   - override_means: Boolean to apply custom pattern-based overrides.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the merged mean parameter values.
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

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns );  
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0, 10.0, 10.0][1:length(u)]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int)
    """
    libgeos_lattice_adjacency_matrix(rows, cols)

    Description:
    Generates a sparse adjacency matrix for a regular 2D lattice using LibGEOS for spatial geometry operations.
    Constructs unit square polygons for each cell and identifies neighbors based on Queen contiguity
    (any shared boundary point or edge).

    Inputs:
    - rows (Int): Number of rows in the lattice grid.
    - cols (Int): Number of columns in the lattice grid.

    Output:
    - W (SparseMatrixCSC{Int, Int}): A binary sparse adjacency matrix of size (rows*cols) x (rows*cols).
    """
    # Create polygons for each cell in the lattice
    polygons = []
    for r in 1:rows, c in 1:cols
        # Define unit square coordinates as nested vectors for LibGEOS compatibility
        coords = [
            [Float64(c-1), Float64(r-1)],
            [Float64(c),   Float64(r-1)],
            [Float64(c),   Float64(r)],
            [Float64(c-1), Float64(r)],
            [Float64(c-1), Float64(r-1)]
        ]
        # Construct LinearRing and then Polygon
        ring = LibGEOS.LinearRing(coords)
        push!(polygons, LibGEOS.Polygon(ring))
    end

    n = length(polygons)
    W = spzeros(Int, n, n)

    # Queen contiguity check
    for i in 1:n
        poly_i = polygons[i]
        for j in (i+1):n
            if LibGEOS.intersects(poly_i, polygons[j])
                W[i, j] = W[j, i] = 1
            end
        end
    end
    return W
end




# --- 1. Custom PC Priors ---
struct PCPriorSigma <: ContinuousUnivariateDistribution
    U::Float64
    alpha::Float64
    lambda::Float64
    function PCPriorSigma(U, alpha)
        return new(U, alpha, -log(alpha) / U)
    end
end

function Distributions.logpdf(d::PCPriorSigma, x::Real)
    x > 0 ? log(d.lambda) - d.lambda * x : -Inf
end

Distributions.rand(rng::AbstractRNG, d::PCPriorSigma) = rand(rng, Exponential(1 / d.lambda))
Distributions.minimum(d::PCPriorSigma) = 0.0
Distributions.maximum(d::PCPriorSigma) = Inf
Bijectors.bijector(d::PCPriorSigma) = Bijectors.exp


function build_laplacian_precision(adj_matrix)
    # Description:
    #   Constructs a GMRF precision matrix (Graph Laplacian) from an adjacency matrix.
    # Inputs:
    #   adj_matrix: Sparse adjacency matrix (W).
    # Outputs:
    #   Sparse precision matrix (Q).

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 




function scale_precision!(Q)
    # Description:
    #   Scales a precision matrix using the geometric mean of non-zero eigenvalues.
    #   Essential for ensuring sigma_sp represents marginal variance in BYM2.
    # Inputs:
    #   Q: Precision matrix to be modified in-place.

    eig_vals = eigvals(Matrix(Q))
    # Filter out near-zero eigenvalues associated with the null space
    valid_eigs = filter(x -> x > 1e-6, eig_vals)
    
    scaling_factor = exp(mean(log.(valid_eigs)))
    
    if Q isa Symmetric
        Q.data ./= scaling_factor
    else
        Q ./= scaling_factor
    end
    
    return Q
end





function build_rw2_precision(n)
    # Description: Builds a second-order random walk (RW2) precision matrix for smoothing.
    # Inputs:
    #   - n: Number of categories or time points.
    # Outputs:
    #   - Sparse precision matrix of size n x n.
    D = spzeros(n - 2, n)
    for i in 1:(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    return D' * D
end

function build_ar1_precision(n, rho, tau)
    # Description: Builds a first-order autoregressive (AR1) precision matrix.
    # Inputs:
    #   - n: Number of time points.
    #   - rho: Correlation coefficient.
    #   - tau: Precision scale.
    # Outputs:
    #   - Sparse precision matrix.
    T = promote_type(typeof(rho), typeof(tau))
    diag_vals = [one(T); fill(one(T) + rho^2, n - 2); one(T)]
    off_diag = fill(-rho, n - 1)
    Q = spdiagm(0 => diag_vals, 1 => off_diag, -1 => off_diag)
    return (tau / (one(T) - rho^2)) * Q
end

function build_cyclic_ar1_precision(n, rho, tau)
    # Description: Builds a cyclic AR1 precision matrix (wrapping last to first).
    # Inputs:
    #   - n: Number of time points.
    #   - rho: Correlation coefficient.
    #   - tau: Precision scale.
    # Outputs:
    #   - Sparse precision matrix.
    T = promote_type(typeof(rho), typeof(tau))
    Q = zeros(T, n, n)
    for i in 1:n
        Q[i, i] = one(T) + rho^2
        prev, nxt = (i == 1 ? n : i - 1), (i == n ? 1 : i + 1)
        Q[i, prev] = -rho
        Q[i, nxt] = -rho
    end
    return (tau / (one(T) - rho^2)) * Q
end


function logpdf_gmrf(x, Q)
    # Description: Calculates the log-probability of a Gaussian Markov Random Field.
    # Inputs:
    #   - x: Vector of values.
    #   - Q: Precision matrix.
    # Outputs:
    #   - Log-likelihood value.
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2pi))
end


 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_coord_tuple=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    # Extract target stats and guard against nothing or scalar values
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end

    if !isnothing(M)
        s_coord_tuple = M.s_coord_tuple
    end
 
    # 1. Handle Categorical/Class Bar Plots
    if effect == :beta_cov
        b_list = get(stats, :beta_cov, nothing)
        isnothing(b_list) && return nothing
        b_stats = b_list isa AbstractVector ? b_list[cov_idx] : b_list
        (isnothing(b_stats) || b_stats isa Real) && return nothing
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)

    elseif effect == :b_class1 || effect == :b_class2
        b_stats = st
        if isnothing(b_stats) || b_stats isa Real; return nothing; end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1],
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    # 2. Handle Temporal Main Effects
    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    # 2a. Handle Seasonal Effects
    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)


    # 3. Handle Spatial, ST, and Deep GP Mean Fields
    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        # Determine the values to map to colors
        values = if hasproperty(st, :mean)
            st.mean
        elseif effect == :spatial_structured
            stats.spatial_structured.mean
        elseif effect == :spatial_unstructured
            stats.spatial_unstructured.mean
        elseif effect == :eta_gp
            haskey(stats, :eta_gp) ? stats.eta_gp.mean : error("eta_gp not found in stats")
        elseif effect == :hidden_layer
            haskey(stats, :h1) ? stats.h1.mean : error("hidden layer h1 not found in stats")
        elseif effect == :predictions_denoised && !isnothing(time_slice)
            stats.predictions_denoised.mean[:, time_slice]
        elseif effect == :predictions_noisy && !isnothing(time_slice)
            stats.predictions_noisy.mean[:, time_slice]
        # elseif effect == :residuals
        #    stats.predictions_noisy.mean[:, isnothing(time_slice) ? 1 : time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        # SAFETY FIX: Plot only as many polygons as we have results for to avoid BoundsError
        n_to_plot = min(length(areal_units.polygons), length(values))

        for i in 1:n_to_plot
            poly_coords = areal_units.polygons[i]
            if length(poly_coords) > 2
                px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
                py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

                if !isempty(px)
                    if (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1]); push!(py, py[1])
                    end

                    val = values[i]
                    StatsPlots.plot!(plt, px, py,
                        seriestype=:shape,
                        fill_z=val,
                        c=:RdYlBu,
                        linecolor=:black,
                        linewidth=0.5,
                        fillalpha=0.8,
                        legend=false
                    )
                end
            end
        end

        if show_pts
            StatsPlots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
                markersize=1, markercolor=:gray, alpha=0.2, label="Observations")
        end

        StatsPlots.scatter!(plt, [c[1] for c in areal_units.centroids], [c[2] for c in areal_units.centroids],
            markersize=2, markercolor=:white, markerstrokecolor=:black, alpha=0.5, label="Centroids")

        return plt
    else
        error("Effect $effect not recognized.")
    end
end




function plot_posterior_vs_prior(model::DynamicPPL.Model, chain::MCMCChains.Chains, param_sym::Symbol; n_prior_samples=1000, title="Posterior vs Prior")
    # Description: Overlays posterior and prior densities for a specific parameter to check learning/shrinkage.
    # Inputs:
    #   - model: Turing model object.
    #   - chain: MCMC sample chain.
    #   - param_sym: Symbol of the parameter to check.
    # Outputs:
    #   - A Plots.Plot object.

    # 1. Extract posterior samples using .data for AxisArray compatibility
    post_samples = vec(chain[param_sym].data)

    # 2. Automated Prior Sampling via Turing
    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    # 3. Visualization
    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end




# --- 1. MODEL UTILITIES ---

function NegativeBinomial2(μ, r)
    # Description: Alternative parametrization of Negative Binomial using mean (μ) and dispersion (r).
    # Inputs:
    #   - μ: Mean.
    #   - r: Size/dispersion parameter.
    # Outputs:
    #   - Distributions.NegativeBinomial object.

    p = r / (r + μ)
    return NegativeBinomial(r, p)
end
  

function get_rff_deep2D_basis(X, m, lengthscale)
    # Description: Generates Random Fourier Feature (RFF) basis for 2D inputs (Spatial/Temporal).
    # Inputs:
    #   - X: Input matrix (N x D).
    #   - m: Number of features.
    #   - lengthscale: Gaussian kernel lengthscale.
    # Outputs:
    #   - N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .*  2pi
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end


function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    N = length(t)
    # Generate random parameters for RFFs.
    # Using a seed ensures consistency within the AD pass.
    Random.seed!(42)
    Omega_samples_float = randn(m)
    Phi_phases_float = rand(m)

    Omega_samples = Omega_samples_float ./ lengthscale
    Phi_phases = Phi_phases_float .* convert(T,  2pi)

    Z = zeros(T, N, m)
    for j in 1:m
        Z[:, j] = convert.(T, sqrt(2/m)) .* cos.(Omega_samples[j] .* t .+ Phi_phases[j])
    end
    return Z
end


function get_rff_seasonal_basis(t, m, freq, lengthscale)
    # Description: Generates RFF-style basis for periodic seasonal components.
    # Inputs:
    #   - t: Time vector.
    #   - m: Number of harmonics.
    #   - freq: Base frequency.
    #   - lengthscale: Smoothness scale.
    # Outputs:
    #   - N x (2*m) feature matrix.
    N = length(t)
    Z = zeros(N, 2*m)
    for j in 1:m
        omega_j =  2pi * j * freq
        Z[:, 2j-1] = cos.(omega_j .* t)
        Z[:, 2j] = sin.(omega_j .* t)
    end
    return Z
end

function bstm_options(M_previous::NamedTuple; kwargs...)
    # Convert NamedTuple to Dict to allow merging
    M_new = Dict{Symbol, Any}(pairs(M_previous))
    
    # Update with new keyword arguments
    for (key, val) in kwargs
        if isnothing(val)
            delete!(M_new, key)
            M_new[key] = nothing
        else
            M_new[key] = val
        end
    end
    # Re-run original bstm_options to ensure all computed fields are correctly updated
    return bstm_options(; M_new...)
end

 function bstm_options(; kwargs...)
    M0 = Dict{Symbol, Any}(kwargs)

    # --- 1. Architectural & Family Settings ---
    model_arch   = get(M0, :model_arch, "univariate")
    model_family = get(M0, :model_family, "gaussian")
    model_space  = get(M0, :model_space, "bym2")
    model_time   = get(M0, :model_time, "ar1")
    model_season = get(M0, :model_season, "none")
    model_st     = get(M0, :model_st, "none")
    noise        = get(M0, :noise, 1e-6)

    # --- 2. Observation & Dimensionality ---
    y_obs      = get(M0, :y_obs, nothing)
    y_N        = get(M0, :y_N, isnothing(y_obs) ? 0 : size(y_obs, 1))
    W          = get(M0, :W, sparse(I(1)))
    s_N        = size(W, 1)

    s_idx      = get(M0, :s_idx, isnothing(y_obs) ? collect(1:s_N) : ones(Int, y_N))
    t_idx      = get(M0, :t_idx, ones(Int, y_N))
    t_N        = get(M0, :t_N, (t_idx isa AbstractVector && !isempty(t_idx) ? maximum(t_idx) : 1))
    u_idx      = get(M0, :u_idx, ones(Int, y_N))
    u_N        = get(M0, :u_N, 12)

    s_coord_tuple = get(M0, :s_coord_tuple, nothing)
    s_coord = isnothing(s_coord_tuple) ? nothing : RowVecs(reduce(hcat, [collect(p) for p in s_coord_tuple])')
    s_coord_unique = !isnothing(s_coord) ? unique(s_coord) : nothing

    st_idx = get(M0, :st_idx, (t_idx .- 1) .* s_N .+ s_idx)

    use_zi = get(M0, :use_zi, false)
    use_sv = get(M0, :use_sv, false)

    # --- 3. Spatial Precision & Advanced Precomputes ---
    s_Q = sparse(I(s_N))
    K_nystrom_proj = nothing
    cluster_assignments = nothing
    deep_gp_input = nothing

    if model_space in ["bym2", "besag", "leroux", "sar", "dag"]
        D_sp   = Diagonal(vec(sum(W, dims=2)))
        Q_raw  = D_sp - W
        eigs   = filter(x -> x > 1e-6, eigvals(Matrix(Q_raw)))
        sf     = isempty(eigs) ? 1.0 : exp(mean(log.(eigs)))
        s_Q    = Symmetric(sparse(Q_raw ./ sf))
    end

    if model_space == "nystrom"
        kernel_nystrom = get(M0, :kernel_nystrom, Matern32Kernel() ∘ ScaleTransform(1.0))
        inducing_pts = get(M0, :inducing_points, nothing)
        if !isnothing(s_coord) && !isnothing(inducing_pts)
            K_nystrom_proj = precompute_nystrom_projection(s_coord, inducing_pts, kernel_nystrom)
        end
    elseif model_space == "deepGP"
        if !isnothing(s_coord_tuple)
            s_coord_mat = reduce(hcat, [collect(p) for p in s_coord_tuple])'
            deep_gp_input = hcat(s_coord_mat, Float64.(t_idx) ./ t_N)
        end
    elseif model_space == "local" || model_space == "mosaic"
        n_clusters = get(M0, :n_clusters, 4)
        if !isnothing(s_coord_unique)
            cl_res = kmeans(reduce(hcat, [collect(p) for p in s_coord_unique])', n_clusters)
            cluster_assignments = cl_res.assignments
        end
    end

    # --- 4. Temporal & Seasonal Precomputes ---
    t_Q = model_time == "ar1" ? build_bstm_ar1_template(t_N).matrix : (model_time == "rw2" ? build_bstm_rw2_template(t_N).matrix : sparse(I(t_N)))
    u_Q = model_season == "ar1" ? build_bstm_ar1_template(u_N).matrix : (model_season == "rw2" ? build_bstm_rw2_template(u_N).matrix : sparse(I(u_N)))

    t_period = get(M0, :t_period, Float64(t_N))
    t_angle = (model_time == "harmonic") ? 2pi .* (1:t_N) ./ t_period : nothing
    u_angle = (model_season == "harmonic") ? 2pi .* (1:u_N) ./ 12.0 : nothing

    t_Z_rff = nothing
    if model_time == "rff"
        m_rff_t = get(M0, :m_rff_t, 10)
        t_ls = get(M0, :t_ls, 0.5)
        t_Z_rff = get_rff_trend_basis(collect(1.0:t_N) ./ t_N, m_rff_t, t_ls)
    end

    # --- 5. Covariate Processing & Random Effect Rules ---
    cov_re_structures = Dict{Symbol, Any}()
    cov_groups = Dict{Symbol, Vector{Int}}()

    if haskey(M0, :cov) && haskey(M0, :re_rules)
        for (c_name, rule) in M0[:re_rules]
            c_sym = Symbol(c_name)
            n_bins = get(get(M0, :cov_discretization, Dict()), c_sym, 10)
            # FIX: Convert c_name to Symbol to safely index NamedArrays
            vals = M0[:cov][:, Symbol(c_name)]

            bin_edges = range(minimum(vals), maximum(vals), length=n_bins+1)
            cov_groups[c_sym] = clamp.(searchsortedfirst.(Ref(bin_edges), vals) .- 1, 1, n_bins)

            if rule == "rw2" || rule == "harmonic"
                D1 = [i == j ? -1.0 : (j == i + 1 ? 1.0 : 0.0) for i in 1:n_bins-1, j in 1:n_bins]
                D2 = [i == j ? -1.0 : (j == i + 1 ? 1.0 : 0.0) for i in 1:n_bins-2, j in 1:n_bins-1]
                cov_re_structures[c_sym] = (D2 * D1)' * (D2 * D1)
            elseif rule == "ar1"
                cov_re_structures[c_sym] = build_ar1_precision_matrix(n_bins, 0.0, 1.0)
            elseif rule == "iid"
                cov_re_structures[c_sym] = Matrix(1.0I, n_bins, n_bins)
            elseif rule == "gp"
                coords = collect(1.0:Float64(n_bins))
                K = kernelmatrix(SqExponentialKernel(), coords) + noise*I
                cov_re_structures[c_sym] = inv(Symmetric(Matrix(K)))
            elseif rule == "none"
                cov_re_structures[c_sym] = Matrix(1e6*I, n_bins, n_bins)
            end
        end
    end

    # --- 6. Transport & Interaction Precomputes ---
    st_diffusion_operator = nothing
    if model_st in ["diffusion", "advection_diffusion"]
        st_diffusion_operator = Matrix(Diagonal(vec(sum(W, dims=2))) - W)
    end

    # --- 7. Final Consolidation ---
    updates = (
        model_arch=model_arch, model_family=model_family, model_space=model_space,
        model_time=model_time, model_season=model_season, model_st=model_st,
        y_N=y_N, s_N=s_N, t_N=t_N, u_N=u_N, noise=noise,
        s_idx=s_idx, t_idx=t_idx, u_idx=u_idx, st_idx=st_idx,
        s_coord=s_coord, s_coord_unique=s_coord_unique, s_Q=s_Q, t_Q=t_Q, u_Q=u_Q,
        t_angle=t_angle, u_angle=u_angle, t_Z_rff=t_Z_rff,
        K_nystrom_proj=K_nystrom_proj, cluster_assignments=cluster_assignments, deep_gp_input=deep_gp_input,
        cov_groups=cov_groups, cov_re_structures=cov_re_structures,
        st_diffusion_operator=st_diffusion_operator,
        use_zi=use_zi, use_sv=use_sv,
        weights=get(M0, :weights, ones(y_N)),
        log_offset=get(M0, :log_offset, zeros(y_N)),
        trials=get(M0, :trials, ones(Int, y_N)),
        fixed=get(M0, :fixed, ones(y_N, 1)),
        fixed_N=get(M0, :fixed_N, size(get(M0, :fixed, ones(y_N, 1)), 2))
    )

    for (k, v) in pairs(updates); M0[k] = v; end
    return NamedTuple(M0)
end


function apply_discretization_logic(vals, rules)
    groups = nothing
    new_vals = nothing

    if rules == 0 || isnothing(rules)
        groups = collect(1:length(vals))
    elseif rules == "unit"
        min_v, max_v = minimum(vals), maximum(vals)
        new_vals = (min_v == max_v) ? zeros(length(vals)) : (vals .- min_v) ./ (max_v - min_v)
        groups = collect(1:length(vals))
    elseif rules == "zscore"
        m, s = Statistics.mean(vals), Statistics.std(vals)
        new_vals = (s ≈ 0.0) ? zeros(length(vals)) : (vals .- m) ./ s
        groups = collect(1:length(vals))
    elseif rules == "log"
        new_vals = log.(vals .+ 1.0 .- minimum(vals))
        groups = collect(1:length(vals))
    elseif rules isa Int
        if rules > 1
            qs = unique(sort(quantile(vals, (0:rules) ./ rules)))
            groups = (length(qs) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(qs, x), vals), 1, length(qs)-1)
        else
            groups = ones(Int, length(vals))
        end
    elseif rules isa AbstractString && startswith(rules, "regular:")
        n = parse(Int, Base.split(rules, ":")[2])
        q025, q975 = quantile(vals, 0.025), quantile(vals, 0.975)
        bins = unique(sort(collect(range(q025, stop=q975, length=n+1))))
        groups = (length(bins) < 2) ? ones(Int, length(vals)) : clamp.(map(x -> searchsortedlast(bins, x), vals), 1, length(bins)-1)
    elseif rules isa AbstractVector
        groups = clamp.(map(x -> searchsortedlast(rules, x) + 1, vals), 1, length(rules) + 1)
    end
    return new_vals, groups
end

function assign_covariate_units(cov_data_base, cov_discretization, re_rules, cov_interactions)
    cov_data_for_processing = deepcopy(cov_data_base)
    cov_groups = Dict{Symbol, Vector{Int}}()
    cov_re_structures = Dict{Symbol, Any}()

    if !isnothing(cov_discretization)
        for cov_name_sym in names(cov_data_base, 1)
            if haskey(cov_discretization, cov_name_sym)
                rules = cov_discretization[cov_name_sym]
                vals = cov_data_for_processing[cov_name_sym, :]
                new_vals, groups = apply_discretization_logic(vals, rules)
                if !isnothing(new_vals); cov_data_for_processing[cov_name_sym, :] = new_vals; end
                if !isnothing(groups)
                    cov_groups[cov_name_sym] = groups
                    if haskey(re_rules, cov_name_sym) && length(unique(groups)) > 1
                        n_bins = length(unique(groups))
                        if re_rules[cov_name_sym] == "rw2"; cov_re_structures[cov_name_sym] = build_bstm_rw2_template(n_bins).matrix
                        elseif re_rules[cov_name_sym] == "ar1"; cov_re_structures[cov_name_sym] = build_bstm_ar1_template(n_bins).matrix
                        end
                    end
                end
            end
        end
    end

    for inter_str in cov_interactions
        parts = Base.split(inter_str, "*")
        if length(parts) == 2
            n1, n2 = Symbol(parts[1]), Symbol(parts[2])
            if n1 in names(cov_data_for_processing, 1) && n2 in names(cov_data_for_processing, 1)
                inter_val = cov_data_for_processing[n1, :] .* cov_data_for_processing[n2, :]
                new_row = NamedArray(inter_val', (Symbol[Symbol(inter_str)], names(cov_data_for_processing, 2)))
                cov_data_for_processing = vcat(cov_data_for_processing, new_row)
                if !isnothing(cov_discretization) && haskey(cov_discretization, Symbol(inter_str))
                    rule_int = cov_discretization[Symbol(inter_str)]
                    iv_vals, iv_groups = apply_discretization_logic(inter_val, rule_int)
                    if !isnothing(iv_vals); cov_data_for_processing[Symbol(inter_str), :] = iv_vals; end
                    if !isnothing(iv_groups)
                        cov_groups[Symbol(inter_str)] = iv_groups
                        if haskey(re_rules, Symbol(inter_str)) && length(unique(iv_groups)) > 1
                            n_bins = length(unique(iv_groups))
                            if re_rules[Symbol(inter_str)] == "rw2"; cov_re_structures[Symbol(inter_str)] = build_bstm_rw2_template(n_bins).matrix
                            elseif re_rules[Symbol(inter_str)] == "ar1"; cov_re_structures[Symbol(inter_str)] = build_bstm_ar1_template(n_bins).matrix
                            end
                        end
                    end
                end
            end
        end
    end
    return cov_data_for_processing, cov_groups, cov_re_structures
end


# -----------------------


function get_chain_names(chain)
    try
        return string.(FlexiChains.parameters(chain))
    catch
        return string.(names(chain))
    end
end


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing)
    y_sig_samples = zeros(N_tot, N_samples)
    for j in 1:N_samples
        local sig_y
        if isnothing(outcome_idx)
            if "y_sigma" in name_strs
                val = chain[:y_sigma][j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif "y_sigma_const" in name_strs
                sig_val = Float64(chain[:y_sigma_const][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        else
            v_key = "y_sigma_k[$outcome_idx]"
            c_key = "y_sigma_const_k[$outcome_idx]"
            if v_key in name_strs
                val = chain[Symbol(v_key)][j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif c_key in name_strs
                sig_val = Float64(chain[Symbol(c_key)][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        end

        # Final flattening and type enforcement to prevent MethodError
        flat_sig = vec(Float64.(collect(sig_y)))
        
        if length(flat_sig) >= N_tot
            y_sig_samples[:, j] = flat_sig[1:N_tot]
        else
            y_sig_samples[1:length(flat_sig), j] = flat_sig
            y_sig_samples[length(flat_sig)+1:end, j] .= flat_sig[end]
        end
    end
    return y_sig_samples
end



function get_params_vector(chain, base_name, len)
    # Robust parameter extraction for FlexiChains/MCMCChains
    N_samples = size(chain, 1)

    # Use FlexiChains.parameters to get names
    names_ch = string.(FlexiChains.parameters(chain))

    # Tier 1: Indexed names [k]
    regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    matched_names = filter(n -> occursin(regex, n), names_ch)

    if !isempty(matched_names)
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))
        res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            val_obj = chain[Symbol(n)]
            raw = hasproperty(val_obj, :data) ? val_obj.data : Array(val_obj)
            # Flatten nested structures if they appear at the index level
            data_fixed = raw isa AbstractVector && eltype(raw) <: AbstractVector ? reduce(vcat, raw) : raw
            res_mat[:, idx] = vec(Float64.(collect(data_fixed)))
        end

        if size(res_mat, 2) == 1 && len > 1
            return repeat(res_mat, 1, len)
        end
        return res_mat
    end

    # Tier 2: Single entity fallback
    if base_name in names_ch
        val_obj = chain[Symbol(base_name)]
        raw_data = hasproperty(val_obj, :data) ? val_obj.data : Array(val_obj)
        if ndims(raw_data) == 3; raw_data = raw_data[:, :, 1]; end

        # Standardize to Matrix [Samples x Params]
        # Robustly handle Matrix{Vector{Float64}} or Vector{Vector{Float64}} by flattening
        if raw_data isa AbstractMatrix && eltype(raw_data) <: AbstractVector
            # Flatten each row of vectors into a single parameter row
            mat_data = reduce(hcat, [reduce(vcat, row) for row in eachrow(raw_data)])'
        elseif raw_data isa AbstractArray && eltype(raw_data) <: AbstractVector
            mat_data = reduce(hcat, vec(raw_data))'
        else
            mat_data = Matrix{Float64}(raw_data)
        end

        if size(mat_data, 2) == len
            return mat_data
        elseif size(mat_data, 1) == len
            return mat_data'
        elseif size(mat_data, 2) == 1
            return repeat(mat_data, 1, len)
        end
    end

    @warn "Parameter '$base_name' not found in chain. Returning zeros for length $len."
    return zeros(Float64, N_samples, len)
end




function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
"""
    generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)

Generates 2D RFF weights W by sampling frequencies from the provided 2D magnitude spectrum.

Args:
    freqs_x: Vector of x-dimension frequencies.
    freqs_y: Vector of y-dimension frequencies.
    magnitude_spectrum: 2D array of magnitude values corresponding to freqs_x, freqs_y.
    M_rff_count: Number of RFF features to generate.

Returns:
    A 2 x M_rff_count matrix for W_fixed.
"""
    # Flatten frequency grids and magnitude spectrum into 1D arrays for sampling
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    # Normalize magnitudes to form a probability distribution
    # Add a small constant to magnitudes before normalization to prevent division by zero for zero probabilities.
    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    # Sample M_rff_count indices based on probabilities
    # StatsBase.sample expects Weights from non-negative numbers
    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    W_fixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        W_fixed[1, i] = all_freqs_x[idx] *  2pi # Scale by  2pi to match RFF convention (often ω'x)
        W_fixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return W_fixed
end
 

# Helper to create AR1 precision matrix
function ar1_precision(n, rho, sigma_e)
    # Get the type of the parameters, which will be ForwardDiff.Dual during AD
    # or Float64 when not differentiating
    T = typeof(rho)
    Q = spzeros(T, n, n) # Explicitly create a sparse matrix that can hold Dual numbers

    # Main diagonal
    Q[1, 1] = one(T) # Use one(T) to ensure type compatibility
    for i in 2:(n - 1)
        Q[i, i] = one(T) + rho^2
    end
    Q[n, n] = one(T)
    # Off-diagonals
    for i in 1:(n - 1)
        Q[i, i + 1] = -rho
        Q[i + 1, i] = -rho
    end
    # Ensure division also uses the correct type, and result type of `inv(sigma_e^2)` matches T
    return (one(T) / sigma_e^2) .* Q
end

# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    n = length(times)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, n, n) # Initialize matrix with this type
    for i in 1:n
        for j in 1:n
            C[i, j] = sigma_e^2 * rho^abs(times[i] - times[j])
        end
    end
    return C
end

# Helper to create AR1 cross-covariance matrix
function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    na = length(times_a)
    nb = length(times_b)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, na, nb) # Initialize matrix with this type
    for i in 1:na
        for j in 1:nb
            C[i, j] = sigma_e^2 * rho^abs(times_a[i] - times_b[j])
        end
    end
    return C
end

# Helper for Kronecker AR1 x Matern Sampling
function kron_ar1_matern_sample(Ns, Nt, unique_s, ls_s, sigma_s, rho_t, sigma_t_noise, noise_vec, noise=1e-4)
    # Spatial Matern 3/2 Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + noise*I )
    Q_s = sparse(inv(K_s))

    # Temporal AR1 Precision
    Q_t = ar1_precision(Nt, rho_t, sigma_t_noise)

    # Kronecker Product Q = Qt ⊗ Qs
    Q_full = Symmetric( kron(Q_t, Q_s) + noise*I)

    # Explicitly ensure symmetry for sparse matrix before Cholesky decomposition
    # Convert to dense Matrix to avoid SparseArrays.CHOLMOD incompatibility with ForwardDiff.Dual
    L_q = cholesky(Symmetric(Matrix(Q_full) + noise*I)) # Increased noise

    # Correctly extract the lower triangular factor for dense Cholesky
    return L_q.L' \ noise_vec
end
 

function prepare_fft_grid(s_coord_tuple, values; grid_res=64, pad_factor=2)
    # 1. Define the bounding box
    xs = [p[1] for p in s_coord_tuple]
    ys = [p[2] for p in s_coord_tuple]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # 2. Map points to a grid
    # Use the length of the shorter input to prevent BoundsError
    n_limit = min(length(s_coord_tuple), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        p = s_coord_tuple[i]
        ix = Int(floor((p[1] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((p[2] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    # 3. Apply Zero-Padding
    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end
 



#####



abstract type AbstractModelArchitecture end

struct DidacticArchitecture <: AbstractModelArchitecture end 
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultioutcomeArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ComplexArchitecture <: AbstractModelArchitecture end
struct SizeStructuredArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end

function get_architecture(model_arch::String)
    if model_arch=="example"
        return DidacticArchitecture()
    elseif model_arch in ["univariate"]
        return UnivariateArchitecture()
    elseif model_arch in ["multivariate", "joint"]
        return MultivariateArchitecture()
    elseif model_arch in ["multioutcome"]
        return MultioutcomeArchitecture()
    elseif model_arch in ["multifidelity"]
        return MultifidelityArchitecture()
    elseif model_arch == "size_structured"
        return SizeStructuredArchitecture()
    elseif model_arch in ["complex"]
        return ComplexArchitecture()
    else
        return UnknownArchitecture()
    end
end

 
abstract type ModelFamily end
struct PoissonFamily <: ModelFamily end
struct GaussianFamily <: ModelFamily end
struct LogNormalFamily <: ModelFamily end
struct BinomialFamily <: ModelFamily end
struct NegativeBinomialFamily <: ModelFamily end

function get_model_family(model_family::String)
    if model_family == "poisson"
        return PoissonFamily()
    elseif model_family == "gaussian"
        return GaussianFamily()
    elseif model_family == "lognormal"
        return LogNormalFamily()
    elseif model_family in ["bernoulli", "binomial"]
        return BinomialFamily()
    elseif model_family == "negbin"
        return NegativeBinomialFamily()
    else
        error("Unknown model_family: $model_family")
    end
end



function reconstruct_posteriors( chain, M, PS; alpha=0.05)
    arch = get_architecture(M.model_arch)
    fam = get_model_family(M.model_family)
    
    return _reconstruct(arch, fam, chain, M, PS, alpha)
end
 

function get_model_parameters(m::DynamicPPL.Model)
    # Directly extract names from a model instance sample as a fallback discovery method
    try
        raw_keys = keys(rand(m))
        return map(raw_keys) do k
            replace(string(k), r"[\(\)\"\:]" => "")
        end
    catch
        return String[]
    end
end

function check_has_parameter(m::DynamicPPL.Model, param_name::String)
    all_p = get_model_parameters(m)
    return any(n -> n == param_name || startswith(n, "$param_name["), all_p)
end
 

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.y_N)

    # Helper to handle parameter access across different chain types
    name_strs = string.(FlexiChains.parameters(chain))

    for j in 1:N_samples
        # Extract volatility (Heteroskedastic vs Homoskedastic)
        local sig_y
        if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        elseif "y_sigma" in name_strs
            sig_y = vec(chain[:y_sigma].data[j])
        elseif "y_sigma_const" in name_strs
            sig_val = Float64(chain[:y_sigma_const].data[j])
            sig_y = fill(sig_val, N_tot)
        else
            sig_y = fill(1.0, N_tot)
        end

        for i in 1:N_tot
            is_obs = i <= M.y_N
            mu_eta = eta[i, j]

            # --- Link Functions ---
            mu = if fam isa PoissonFamily || fam isa NegativeBinomialFamily
                clamp(exp(mu_eta), 1e-10, 1e9) 
            elseif fam isa BinomialFamily
                logistic(mu_eta)
            elseif fam isa LogNormalFamily
                mu_eta 
            else
                mu_eta
            end

            denoised[i, j] = mu

            # --- Likelihood Calculation (Training Data Only) ---
            if is_obs
                y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                y_val = y_vals_src[i]
                log_lik[j, i] = if fam isa PoissonFamily; logpdf(Poisson(mu), y_val)
                               elseif fam isa GaussianFamily; logpdf(Normal(mu, sig_y[i]), y_val)
                               elseif fam isa BinomialFamily; logpdf(Binomial(M.trials[i], mu), y_val)
                               elseif fam isa LogNormalFamily; logpdf(LogNormal(mu, sig_y[i]), y_val)
                               elseif fam isa NegativeBinomialFamily
                                   r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0
                                   prob = r_val / (r_val + mu)
                                   logpdf(NegativeBinomial(r_val, prob), y_val)
                               else 0.0 end
            end

            # --- Posterior Predictive Sampling ---
            noisy[i, j] = if fam isa GaussianFamily; mu + randn() * sig_y[i]
                          elseif fam isa LogNormalFamily; rand(LogNormal(mu, sig_y[i]))
                          elseif fam isa PoissonFamily; rand(Poisson(mu))
                          elseif fam isa BinomialFamily
                               n_trials = is_obs ? M.trials[i] : 1 
                               rand(Binomial(n_trials, mu))
                          elseif fam isa NegativeBinomialFamily
                               r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0
                               rand(NegativeBinomial(r_val, r_val / (r_val + mu)))
                          else mu end
        end
    end
    return denoised, noisy, log_lik
end


function _compute_waic(log_lik)
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
    # Identify all categorical covariate groups present in the chain
    # Matches patterns like "beta_cov[1]", "beta_cov[2]", etc.
    cov_matches = unique(map(m -> m.captures[1], filter(!isnothing, match.(r"beta_cov\[(\d+)\]", all_names))))
    
    if isempty(cov_matches)
        return nothing
    end

    # Parse indices and sort to ensure sequential processing
    cov_indices = sort(parse.(Int, cov_matches))
    
    results = []
    for k in cov_indices
        base_name = "beta_cov[$k]"
        # Use the robust get_params_vector helper to extract the full vector for this covariate group
        raw_vals = get_params_vector(chain, base_name, M.N_cat)
        
        if !all(raw_vals .== 0) # Only process if data was actually found
            # Reshape for summarize_array (N_categories x 1 x N_samples)
            summ = summarize_array(reshape(raw_vals', M.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end

function create_prediction_surface(basis_df, observations_df, au; fixed_df=nothing, lambda_s=2.0, lambda_t=1.0, max_iters=5)
    # 1. Initialization and Automatic Covariate Detection
    mergeon = hasproperty(basis_df, :u_idx) && hasproperty(observations_df, :u_idx) ? [:s_idx, :t_idx, :u_idx] : [:s_idx, :t_idx]

    # Determine covariates: columns in observations_df that are NOT merge keys
    all_covs = setdiff(propertynames(observations_df), mergeon)

    # Join datasets
    surface = leftjoin(basis_df, observations_df, on = mergeon)

    # FIX: Explicitly convert EVERY column to Float64/Missing to prevent InexactError during assignment
    for c in propertynames(surface)
        surface[!, c] = convert(Vector{Union{Float64, Missing}}, collect(surface[!, c]))
    end

    # Merge Fixed Effects
    if !isnothing(fixed_df)
        join_cols = intersect(mergeon, propertynames(fixed_df))
        if !isempty(join_cols)
            surface = leftjoin(surface, fixed_df, on = join_cols, makeunique=true)
            for c in propertynames(surface)
                 surface[!, c] = convert(Vector{Union{Float64, Missing}}, collect(surface[!, c]))
            end
        end
    end

    if isempty(all_covs)
        return surface
    end

    centroids = au.centroids
    n_units = length(centroids)
    W = au.W

    # 2. Precompute Weights
    dist_mat_s = [sqrt(sum((c1 .- c2).^2)) for c1 in centroids, c2 in centroids]
    weight_mat_s = exp.(-dist_mat_s.^2 ./ (2 * lambda_s^2)) .* (W + I)
    for i in 1:n_units
        s_row = sum(weight_mat_s[i, :])
        if s_row > 1e-9; weight_mat_s[i, :] ./= s_row; end
    end

    # 3. Iterative Spatiotemporal Imputation
    for iter in 1:max_iters
        missing_before = sum(ismissing.(Matrix(surface[:, all_covs])))

        for c in all_covs
            group_cols = hasproperty(surface, :u_idx) ? [:u_idx] : []
            for sub_df in groupby(surface, group_cols)
                for i in 1:nrow(sub_df)
                    if ismissing(sub_df[i, c]) || isnan(sub_df[i, c])
                        curr_s = Int(round(sub_df[i, :s_idx]))
                        curr_t = sub_df[i, :t_idx]

                        # Spatial contribution
                        spatial_mask = (sub_df.t_idx .== curr_t)
                        spatial_neighbors = sub_df[spatial_mask, :]
                        s_vals = filter(x -> !ismissing(x[2]) && !isnan(x[2]), collect(zip(spatial_neighbors.s_idx, spatial_neighbors[!, c])))

                        val_s, weight_s_sum = 0.0, 0.0
                        for (nb_s, nb_val) in s_vals
                            idx_s = Int(round(nb_s))
                            if idx_s >= 1 && idx_s <= n_units
                                w = weight_mat_s[curr_s, idx_s]
                                val_s += w * Float64(nb_val)
                                weight_s_sum += w
                            end
                        end

                        # Temporal contribution
                        temporal_mask = (sub_df.s_idx .== sub_df[i, :s_idx])
                        temporal_neighbors = sub_df[temporal_mask, :]
                        t_vals = filter(x -> !ismissing(x[2]) && !isnan(x[2]), collect(zip(temporal_neighbors.t_idx, temporal_neighbors[!, c])))

                        val_t, weight_t_sum = 0.0, 0.0
                        for (nb_t, nb_val) in t_vals
                            w = exp(-(curr_t - nb_t)^2 / (2 * lambda_t^2))
                            val_t += w * Float64(nb_val)
                            weight_t_sum += w
                        end

                        if (weight_s_sum + weight_t_sum) > 1e-6
                            sub_df[i, c] = (val_s + val_t) / (weight_s_sum + weight_t_sum)
                        end
                    end
                end
            end
        end

        missing_after = sum(ismissing.(Matrix(surface[:, all_covs])))
        if missing_after == 0 || missing_after == missing_before; break; end
    end

    # 4. Final Fallback
    for c in all_covs
        valid_data = filter(x -> !ismissing(x) && !isnan(x), surface[!, c])
        m_val = isempty(valid_data) ? 0.0 : Float64(median(valid_data))
        surface[!, c] = map(x -> (ismissing(x) || isnan(x)) ? m_val : Float64(x), surface[!, c])
    end

    # 5. POST-PROCESSING: Convert _idx columns back to Int64
    for c in propertynames(surface)
        if endswith(string(c), "_idx")
            surface[!, c] = map(x -> ismissing(x) ? missing : Int64(round(x)), surface[!, c])
        end
    end

    return surface
end


function _reconstruct(arch::DidacticArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
 

    name_strs = string.(FlexiChains.parameters(chain))


    s_eta_tot = zeros(N_tot, N_samples)
    s_eta_obs_denoised = zeros(M.y_N, N_samples)
    y_sigma_samples = _extract_volatility(chain, name_strs, N_tot, N_samples)

    # Check for Sparse GP parameters
     
    if M.model_space == "sparseGP"
        
        # "u_latent" in name_strs && haskey(M, :t_inducing)
        
        # --- Sparse GP Reconstruction Branch ---
        n_u = size(M.s_inducing, 2) * size(M.t_inducing, 2)
        u_latent_samples = get_params_vector(chain, "u_latent", n_u)
        st_sigma_samples = vec(get_params_vector(chain, "st_sigma", 1))
        ls_s_samples = vec(get_params_vector(chain, "ls_s", 1))
        ls_t_samples = vec(get_params_vector(chain, "ls_t", 1))

        for s in 1:N_samples
            # Reconstruct Kernel and GP
            k = (st_sigma_samples[s]^2) * (SqExponentialKernel() ∘ ScaleTransform(inv(ls_s_samples[s]))) ⊗ (SqExponentialKernel() ∘ ScaleTransform(inv(ls_t_samples[s])))
            f = GP(k)
            
            X_ind = ColVecs(vcat(M.s_inducing', M.t_inducing'))
            f_u = f(X_ind, 1e-6)
            f_cond = condition(f_u, vec(u_latent_samples[s, :]))

            # Project to training locations
            X_obs = ColVecs(vcat(M.s_coord', M.t_coord'))[M.st_idx]
            s_eta_tot[1:M.y_N, s] = mean(f_cond(X_obs))
            s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]

            # Project to prediction locations
            if N_PS > 0
                X_ps = ColVecs(vcat(PS.s_coord', PS.t_coord'))[PS.st_idx]
                s_eta_tot[M.y_N+1:end, s] = mean(f_cond(X_ps))
            end
        end
    
    elseif M.model_space == "kriging_simple"
    
        # 1. Parameter Extraction for GP components
        s_eta_tot = zeros(M.y_N, N_samples)
        for s in 1:N_samples
            s_eta_tot[:, s] = vec(chain[:f_latent_spatial_all].data[s])
        end

        y_sigma_samples = zeros(N_tot, N_samples)
        for s in 1:N_samples
            sigmas = vec(chain[:obs_std_dev].data[s])
            y_sigma_samples[1:M.y_N, s] = sigmas[M.t_idx]
        end

        # 2. Linear Predictor (eta)
        for s in 1:N_samples
            s_eta_tot[:, s] .+= chain[:mu_global].data[s]  
        end

 
    else
        # --- Standard Latent Extraction Branch ---
        for s in 1:N_samples
            if "s_eta_raw" in name_strs
                full_latent = vec(chain[:s_eta_raw].data[s])
                s_eta_tot[1:M.y_N, s] = full_latent[M.st_idx]
                s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]
            elseif "eta_joint" in name_strs
                s_eta_tot[1:M.y_N, s] = vec(chain[:eta_joint].data[s])
                s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]
            elseif "s_eta" in name_strs
                s_eta_tot[1:M.y_N, s] = vec(chain[:s_eta].data[s])
                s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]
            elseif "eta_spatial_combined" in name_strs
                s_eta_tot[1:M.y_N, s] = vec(chain[:eta_spatial_combined].data[s])
                s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]
            elseif "f_gp" in name_strs
                s_eta_tot[1:M.y_N, s] = vec(chain[:f_gp].data[s])
                s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]
            elseif "f_st" in name_strs
                s_eta_tot[1:M.y_N, s] = vec(chain[:f_st].data[s])
                s_eta_obs_denoised[:, s] = s_eta_tot[1:M.y_N, s]
            end
        end
    end 

    eta = zeros(N_tot, N_samples)
    for s in 1:N_samples
        eta[:, s] = M.log_offset .+ s_eta_tot[:, s]
        if "c_beta" in name_strs
             for k in 1:M.cov_N
                 beta_k = vec(chain[Symbol("c_beta[$k]")].data[s])
                 eta[1:M.y_N, s] .+= beta_k[M.cov_indices[:, k]]
             end
        end
    end

    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)
 
    return (
        spatial_structured = summarize_array(reshape(s_eta_obs_denoised, M.y_N, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eta_tot[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(zeros(M.t_N, N_samples), M.t_N, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(zeros(M.u_N, N_samples), M.u_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = nothing,
        covariate_effects = nothing,
        predictions_observed = summarize_array(reshape(denoised_mu[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik_mat[:, 1:M.y_N]),
        family = fam,
        arch = arch
    )
end


 

function _reconstruct(arch::MultioutcomeArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot_points = M.y_N + N_PS
    
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # Storage for outcome-wise summaries aligned with _reconstruct_template output
    outcome_results = Vector{Any}(undef, M.outcomes_N)

    # Global components
    phi_zi_samples = "phi_zi" in name_strs ? vec(chain[:phi_zi].data) : zeros(N_samples)
    
    for k in 1:M.outcomes_N
        # --- 1. MANIFOLD RECOVERY FOR OUTCOME K ---
        y_sigma_samples_k = _extract_volatility(chain, name_strs, N_tot_points, N_samples, k)

        s_eta_tot = zeros(N_tot_points, N_samples)
        s_eta_obs_denoised = zeros(M.y_N, N_samples)
        t_eta_tot = zeros(M.t_N, N_samples)
        u_eta_tot = zeros(M.u_N, N_samples)
        st_eta_tot = zeros(M.s_N, M.t_N, N_samples)
        fixed_effects_samples_k = zeros(M.fixed_N, N_samples)
        
        # Extract outcome-specific scales
        s_sigma_k = vec(chain[Symbol("s_sigma[$k]")].data)
        t_sigma_k = vec(chain[Symbol("t_sigma[$k]")].data)
        st_sigma_k = "st_sigma" in name_strs ? vec(chain[Symbol("st_sigma[$k]")].data) : ones(N_samples)
        u_sigma_k = "u_sigma" in name_strs ? vec(chain[Symbol("u_sigma[$k]")].data) : zeros(N_samples)
        
        for s in 1:N_samples
            # --- A. SPATIAL MANIFOLD AUDIT (All 14 Structures) ---
            local field_structured_k = zeros(M.s_N)
            local field_noisy_k = zeros(M.s_N)

            if M.model_space == "bym2"
                rho_k = "s_rho" in name_strs ? clamp(chain[Symbol("s_rho[$k]")].data[s], 0.0, 1.0) : 1.0
                icar_k = vec(chain[Symbol("s_icar_k[$k]")].data[s])
                iid_k = vec(chain[Symbol("s_iid_k[$k]")].data[s])
                field_structured_k = sqrt(rho_k) .* icar_k .* s_sigma_k[s]
                field_noisy_k = field_structured_k .+ (sqrt(1 - rho_k) .* iid_k .* s_sigma_k[s])
            elseif M.model_space in ["besag", "icar"]
                field_structured_k = vec(chain[Symbol("s_icar_k[$k]")].data[s]) .* s_sigma_k[s]
                field_noisy_k = field_structured_k
            elseif M.model_space == "sar"
                field_noisy_k = vec(chain[Symbol("s_sar_raw_k[$k]")].data[s]) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "bgcn"
                D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
                W_norm = D_inv_sqrt * M.W * D_inv_sqrt
                field_noisy_k = (W_norm * vec(chain[Symbol("s_icar_k[$k]")].data[s])) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "dag"
                field_noisy_k = vec(chain[Symbol("s_dag_raw_k[$k]")].data[s]) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "local"
                mu_cl = vec(chain[Symbol("mu_clusters_k[$k]")].data[s])
                field_structured_k = mu_cl[M.cluster_assignments]
                field_noisy_k = field_structured_k .+ (vec(chain[Symbol("s_eta_raw_k[$k]")].data[s]) .* s_sigma_k[s])
            elseif M.model_space == "mosaic"
                mu_loc = vec(chain[Symbol("mu_local_k[$k]")].data[s])
                field_structured_k = mu_loc[M.cluster_assignments] .* s_sigma_k[s]
                field_noisy_k = field_structured_k
            elseif M.model_space == "warping"
                w_warp = vec(chain[Symbol("w_warp_k[$k]")].data[s])
                warp_proj = (M.s_coord * M.W_fixed) .+ M.b_fixed'
                field_structured_k = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp) .* s_sigma_k[s]
                field_noisy_k = field_structured_k
            elseif M.model_space == "fft"
                field_noisy_k = (M.s_Q \ vec(chain[Symbol("s_spectral_raw_k[$k]")].data[s])) .* s_sigma_k[s]
                field_structured_k = field_noisy_k
            elseif M.model_space == "iid"
                field_noisy_k = vec(chain[Symbol("s_iid_k[$k]")].data[s]) .* s_sigma_k[s]
                field_structured_k = zeros(M.s_N)
            else # Catch-all for RFF/GP/SVC which map directly to s_eta_scaled if indexed
                field_noisy_k = zeros(M.s_N)
                field_structured_k = zeros(M.s_N)
            end

            # Map to Obs + PS
            s_idx_obs = M.s_idx isa AbstractVector ? M.s_idx : fill(M.s_idx[1], M.y_N)
            s_eta_tot[1:M.y_N, s] = field_noisy_k[s_idx_obs]
            s_eta_obs_denoised[:, s] = field_structured_k[s_idx_obs]
            if N_PS > 0
                s_idx_ps = PS.s_idx isa AbstractVector ? PS.s_idx : fill(PS.s_idx[1], N_PS)
                s_eta_tot[(M.y_N+1):end, s] = field_noisy_k[s_idx_ps]
            end

            # --- B. TEMPORAL FIELD ---
            if M.model_time == "ar1"
                t_eta_tot[:, s] = vec(chain[Symbol("t_raw_k[$k]")].data[s]) .* t_sigma_k[s]
            elseif M.model_time == "rw2"
                t_eta_tot[:, s] = vec(chain[Symbol("t_raw_k[$k]")].data[s]) # Sigma built into precision for RW2
            elseif M.model_time == "gp"
                t_eta_tot[:, s] = vec(chain[Symbol("t_gp_k[$k]")].data[s])
            elseif M.model_time == "harmonic"
                t_a, t_b = chain[Symbol("t_alpha_k[$k]")].data[s], chain[Symbol("t_beta_k[$k]")].data[s]
                t_eta_tot[:, s] = (t_a .* sin.(M.t_angle) .+ t_b .* cos.(M.t_angle)) .* t_sigma_k[s]
            end

            # --- C. SEASONAL FIELD ---
            if M.model_season != "none"
                if M.model_season == "harmonic"
                    u_a, u_b = chain[Symbol("u_alpha_k[$k]")].data[s], chain[Symbol("u_beta_k[$k]")].data[s]
                    u_steps = 1:M.u_N
                    u_eta_tot[:, s] = (u_a .* sin.(2π .* u_steps ./ 12.0) .+ u_b .* cos.(2π .* u_steps ./ 12.0)) .* u_sigma_k[s]
                else
                    u_eta_tot[:, s] = vec(chain[Symbol("u_raw_k[$k]")].data[s]) .* u_sigma_k[s]
                end
            end

            # --- D. SPACE-TIME INTERACTION ---
            if M.model_st != "none"
                st_raw_k = vec(chain[Symbol("st_raw_k[$k]")].data[s])
                st_eta_tot[:, :, s] = reshape(st_raw_k .* st_sigma_k[s], M.s_N, M.t_N)
            end
        end

        # --- 2. PREDICTOR ASSEMBLY ---
        eta_k = zeros(N_tot_points, N_samples)
        for s in 1:N_samples
            st_mat_k = st_eta_tot[:, :, s]
            for i in 1:N_tot_points
                is_obs = i <= M.y_N
                src = is_obs ? M : PS
                idx_adj = is_obs ? i : i - M.y_N
                
                a = src.s_idx isa AbstractVector ? src.s_idx[idx_adj] : src.s_idx
                t = src.t_idx isa AbstractVector ? src.t_idx[idx_adj] : src.t_idx
                u = src.u_idx isa AbstractVector ? src.u_idx[idx_adj] : src.u_idx
                
                val = (is_obs ? M.log_offset[i, k] : 0.0) + s_eta_tot[i, s] + t_eta_tot[t, s] + st_mat_k[a, t]
                if M.model_season != "none"; val += u_eta_tot[u, s]; end
                
                # Fixed Effects Logic for Multivariate
                if M.fixed_N > 0
                    p_beta_k = [chain[Symbol("d_beta_k[$k][$d]")].data[s] for d in 1:M.fixed_N]
                    fixed_effects_samples_k[:, s] = p_beta_k
                    val += dot(p_beta_k, is_obs ? M.fixed[i, :] : PS.fixed[idx_adj, :])
                end
                
                eta_k[i, s] = val
            end
        end

        # --- 3. SUMMARIES PER OUTCOME ---
        denoised_mu, noisy_y, log_lik = _process_ll_and_predictions(fam, eta_k, chain, M, N_tot_points, N_samples, y_sigma_samples_k, M.y_obs[:, k])
        
        outcome_results[k] = (
            spatial_structured = summarize_array(reshape(s_eta_obs_denoised, M.y_N, 1, N_samples); alpha=alpha),
            spatial_noisy = summarize_array(reshape(s_eta_tot[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(t_eta_tot, M.t_N, 1, N_samples); alpha=alpha),
            seasonal = summarize_array(reshape(u_eta_tot, M.u_N, 1, N_samples); alpha=alpha),
            volatility = summarize_array(reshape(y_sigma_samples_k, N_tot_points, 1, N_samples); alpha=alpha),
            fixed_effects = M.fixed_N > 0 ? summarize_array(reshape(fixed_effects_samples_k, M.fixed_N, 1, N_samples); alpha=alpha) : nothing,
            predictions_observed = summarize_array(reshape(denoised_mu[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            waic = _compute_waic(log_lik[:, 1:M.y_N]),
            family = fam
        )
    end

    return (
        outcomes = outcome_results,
        phi_zi = summarize_array(reshape(phi_zi_samples, 1, 1, N_samples); alpha=alpha),
        arch = arch
    )
end


function _reconstruct(arch::MultivariateArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    # Description: Reconstructs factor-based multivariate spatiotemporal fields.
    # Maps latent factor scores back to outcome space via the learned PCA loadings (Householder).

    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)

    N_factors = get(M, :N_factors, 2)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # 1. PCA & Loadings Recovery
    U_samples = zeros(M.outcomes_N, N_factors, N_samples)
    Kmat_samples = zeros(M.outcomes_N, M.outcomes_N, N_samples)
    nvh = Int(M.outcomes_N * N_factors - N_factors * (N_factors - 1) / 2)
    v_all = get_params_vector(chain, "v", nvh)

    for s in 1:N_samples
        pca_sd = [chain[Symbol("pca_sd[$k]")].data[s] for k in 1:N_factors]
        pca_pdef_sd = chain[:pca_pdef_sd].data[s]
        Kmat, _, U = householder_transform(v_all[s, :], M.outcomes_N, N_factors, M.ltri, pca_sd, pca_pdef_sd, M.noise)
        U_samples[:, :, s] = U
        Kmat_samples[:, :, s] = Kmat
    end

    # 2. Factor-wise Field Recovery
    factor_fields = zeros(N_tot, N_factors, N_samples)
    for k in 1:N_factors
        for s in 1:N_samples
            # Spatial Component
            s_sigma_k = chain[Symbol("s_sigma_k[$k]")].data[s]
            local field_k = zeros(M.s_N)
            if M.model_space == "bym2"
                rho_k = clamp(chain[Symbol("s_rho_k[$k]")].data[s], 0.0, 1.0)
                field_k = s_sigma_k .* (sqrt(rho_k) .* vec(chain[Symbol("s_icar_k[$k]")].data[s]) .+ 
                          sqrt(1 - rho_k) .* vec(chain[Symbol("s_iid_k[$k]")].data[s]))
            elseif M.model_space == "besag"
                field_k = vec(chain[Symbol("s_icar_k[$k]")].data[s]) .* s_sigma_k
            elseif M.model_space == "sar"
                field_k = vec(chain[Symbol("s_eta_k[$k]")].data[s])
            end

            # Temporal Component
            t_sigma_k = chain[Symbol("t_sigma_k[$k]")].data[s]
            local trend_k = zeros(M.t_N)
            if M.model_time == "ar1"
                trend_k = vec(chain[Symbol("t_raw_k[$k]")].data[s]) .* t_sigma_k
            elseif M.model_time == "rw2"
                trend_k = vec(chain[Symbol("t_eta_k[$k]")].data[s])
            end

            # Assembly
            for i in 1:N_tot
                is_obs = i <= M.y_N
                src = is_obs ? M : PS
                idx_adj = is_obs ? i : i - M.y_N
                a = src.s_idx isa AbstractVector ? src.s_idx[idx_adj] : src.s_idx
                t = src.t_idx isa AbstractVector ? src.t_idx[idx_adj] : src.t_idx
                factor_fields[i, k, s] = field_k[a] + trend_k[t]
            end
        end
    end

    # 3. Outcome Predictor Synthesis
    eta = zeros(N_tot, M.outcomes_N, N_samples)
    c_eta_all = "c_eta" in name_strs ? get_params_vector(chain, "c_eta", M.N_cat)' : zeros(M.N_cat, N_samples)
    
    fixed_effects_samples = zeros(M.fixed_N, M.outcomes_N, N_samples)
    if M.fixed_N > 0
        for k in 1:M.outcomes_N, d in 1:M.fixed_N
            p_name = Symbol("d_beta[$d,$k]")
            if p_name in Symbol.(name_strs)
                for s in 1:N_samples; fixed_effects_samples[d, k, s] = chain[p_name].data[s]; end
            end
        end
    end

    for s in 1:N_samples
        mu_proj = factor_fields[:, :, s] * U_samples[:, :, s]'
        for i in 1:N_tot
            is_obs = i <= M.y_N
            idx_adj = is_obs ? i : i - M.y_N
            off = is_obs ? M.log_offset[idx_adj, :] : zeros(M.outcomes_N)
            for k in 1:M.outcomes_N
                val = off[k] + mu_proj[i, k]
                if "c_eta" in name_strs
                    val += c_eta_all[is_obs ? M.cov_indices[idx_adj, 1] : PS.cov_indices[idx_adj, 1], s]
                end
                if M.fixed_N > 0
                    fixed_mat = is_obs ? M.fixed : PS.fixed
                    for d in 1:M.fixed_N
                        dm_val = fixed_mat isa AbstractMatrix ? fixed_mat[idx_adj, d] : (fixed_mat isa AbstractVector ? fixed_mat[idx_adj] : fixed_mat)
                        val += fixed_effects_samples[d, k, s] * dm_val
                    end
                end
                eta[i, k, s] = val
            end
        end
    end

    # 4. Summaries
    outcome_results = Vector{Any}(undef, M.outcomes_N)
    for k in 1:M.outcomes_N
        y_sig_k = sqrt.(Kmat_samples[k, k, :])
        y_sigma_samples_k = repeat(y_sig_k', N_tot, 1)
        denoised_k, noisy_k, log_lik_k = _process_ll_and_predictions(fam, eta[:, k, :], chain, M, N_tot, N_samples, y_sigma_samples_k, M.y_obs[:, k])
        outcome_results[k] = (
            spatial_noisy = summarize_array(reshape(eta[1:M.y_N, k, :], M.y_N, 1, N_samples); alpha=alpha),
            volatility = summarize_array(reshape(y_sigma_samples_k, N_tot, 1, N_samples); alpha=alpha),
            fixed_effects = M.fixed_N > 0 ? summarize_array(reshape(fixed_effects_samples[:, k, :], M.fixed_N, 1, N_samples); alpha=alpha) : nothing,
            predictions_observed = summarize_array(reshape(denoised_k[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
            predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_k[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
            waic = _compute_waic(log_lik_k), family = fam
        )
    end
    return (outcomes = outcome_results, loadings = summarize_array(U_samples; alpha=alpha), arch = arch)
end




function _reconstruct(arch::MultifidelityArchitecture, fam::ModelFamily, chain, M, PS, alpha)

    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # 1. MANIFOLD RECOVERY (Latent Field Synthesis)
    s_eta_tot = zeros(N_tot, N_samples)
    s_eta_obs_denoised = zeros(M.y_N, N_samples)
    s_eta_obs_noisy = zeros(M.y_N, N_samples)
    t_eta_tot = zeros(M.t_N, N_samples)
    u_eta_tot = zeros(M.u_N, N_samples)

    # Multi-fidelity specific latent fields
    z_latent_samples = zeros(size(M.z_obs, 1), N_samples)
    w_latent_samples = zeros(size(M.w_obs, 1), 3, N_samples)

    # Extract Volatility Surface
    y_sigma_samples = _extract_volatility(chain, name_strs, N_tot, N_samples)

    # Extract covariate effects
    c_eta_all = zeros(M.N_cat, N_samples)
    if "c_eta" in name_strs
        for s in 1:N_samples
            c_eta_all[:, s] = vec(chain[:c_eta].data[s])
        end
    end

    # Extract fixed effects
    fixed_effects_samples = zeros(M.fixed_N, N_samples)
    if M.fixed_N > 0
        for d in 1:M.fixed_N
            p_name = Symbol("d_beta[$d]")
            if p_name in Symbol.(name_strs)
                for s in 1:N_samples
                    fixed_effects_samples[d, s] = chain[p_name].data[s]
                end
            end
        end
    end

    for s in 1:N_samples
        # Spatial/Temporal Recovery
        s_sigma = "s_sigma" in name_strs ? chain[:s_sigma].data[s] : 1.0
        s_rho = "s_rho" in name_strs ? clamp(chain[:s_rho].data[s], 0.0, 1.0) : 1.0
        s_icar = vec(chain[:s_icar].data[s])
        s_iid = vec(chain[:s_iid].data[s])
        field_structured = s_sigma .* sqrt(s_rho) .* s_icar
        field_noisy = field_structured .+ (s_sigma .* sqrt(1 - s_rho) .* s_iid)

        t_sigma = "t_sigma" in name_strs ? chain[:t_sigma].data[s] : 1.0
        t_raw = vec(chain[:t_raw].data[s])
        field_temporal = t_raw .* t_sigma

        # RFF Latent Fields
        z_latent_samples[:, s] = vec(chain[:z_latent].data[s])
        for k in 1:3
            w_latent_samples[:, k, s] = vec(chain[Symbol("w_latent_k[$k]")].data[s])
        end

        # Map to total grid
        s_eta_tot[1:M.y_N, s] = field_noisy[M.s_idx]
        if N_PS > 0 && !isnothing(PS)
            s_eta_tot[(M.y_N+1):end, s] = field_noisy[PS.s_idx]
        end
        s_eta_obs_denoised[:, s] = field_structured[M.s_idx]
        s_eta_obs_noisy[:, s] = field_noisy[M.s_idx]
        t_eta_tot[:, s] = field_temporal

        # Seasonal (if present)
        if "u_raw" in name_strs
            u_eta_tot[:, s] = vec(chain[:u_raw].data[s]) .* ("u_sigma" in name_strs ? chain[:u_sigma].data[s] : 1.0)
        end
    end

    # 2. PREDICTOR ASSEMBLY
    eta = zeros(N_tot, N_samples)
    z_beta_eta = vec(chain[:z_beta_eta].data)
    w_beta_eta = [vec(chain[Symbol("w_beta_eta[$k]")].data) for k in 1:3]

    for s in 1:N_samples
        for i in 1:N_tot
            is_obs = i <= M.y_N
            idx_adj = is_obs ? i : i - M.y_N
            src = is_obs ? M : PS

            # src can be nothing if i > M.y_N and PS is nothing, though N_tot implies PS exists if i > M.y_N
            a, t, u = src.s_idx[idx_adj], src.t_idx[idx_adj], src.u_idx[idx_adj]
            off = is_obs ? M.log_offset[i] : 0.0

            val = off + s_eta_tot[i, s] + t_eta_tot[t, s] + u_eta_tot[u, s]
            val += z_latent_samples[idx_adj, s] * z_beta_eta[s]
            for k in 1:3
                val += w_latent_samples[idx_adj, k, s] * w_beta_eta[k][s]
            end
            eta[i, s] = val
        end
    end

    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)

    return (
        spatial_structured = summarize_array(reshape(s_eta_obs_denoised, M.y_N, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eta_obs_noisy, M.y_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eta_tot, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(u_eta_tot, M.u_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = summarize_array(reshape(fixed_effects_samples, M.fixed_N, 1, N_samples); alpha=alpha),
        covariate_effects = summarize_array(reshape(c_eta_all, M.N_cat, 1, N_samples); alpha=alpha),
        predictions_observed = summarize_array(reshape(denoised_mu[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        z_latent = summarize_array(reshape(z_latent_samples, size(M.z_obs, 1), 1, N_samples); alpha=alpha),
        w_latent = [summarize_array(reshape(w_latent_samples[:, k, :], size(M.w_obs, 1), 1, N_samples); alpha=alpha) for k in 1:3],
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik_mat[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end


function _reconstruct000(arch::UnivariateArchitecture, fam::ModelFamily, chain, M, PS, alpha)
  
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)

    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    # 1. MANIFOLD RECOVERY
    s_eta_tot = zeros(N_tot, N_samples)
    s_eta_obs_denoised = zeros(M.y_N, N_samples)
    s_eta_obs_noisy = zeros(M.y_N, N_samples)
    t_eta_tot = zeros(M.t_N, N_samples)
    u_eta_tot = zeros(get(M, :u_N, 1), N_samples)
    st_eta_tot = zeros(M.s_N, M.t_N, N_samples)

        
    # Transport Logic Flag
    is_transport = get(M, :model_transport, "none") != "none"

    # Transport-specific parameters
    st_rho_persist_samples = is_transport ? get_params_vector(chain, "st_rho", M.s_N) : zeros(N_samples, M.s_N)
    st_eta_z_samples = is_transport ? get_params_vector(chain, "st_eta_z", M.s_N * M.t_N) : zeros(N_samples, M.s_N * M.t_N)
    st_sigma_transport_samples = is_transport ? vec(get_params_vector(chain, "st_sigma", 1)) : zeros(N_samples) # Transport innovation volatility (re-uses st_sigma name)
    st_diffusion_samples = is_transport ? vec(get_params_vector(chain, "st_diffusion", 1)) : zeros(N_samples)
    st_advection_samples = is_transport ? vec(get_params_vector(chain, "st_advection", 1)) : zeros(N_samples)
 
    y_sigma_samples = _extract_volatility(chain, name_strs, N_tot, N_samples)
    c_eta_all = "c_eta" in name_strs ? get_params_vector(chain, "c_eta", M.N_cat)' : zeros(M.N_cat, N_samples)

    if "c_eta" in name_strs
        for s in 1:N_samples
            c_eta_all[:, s] = vec(chain[:c_eta].data[s])
        end
    end

    fixed_effects_samples = M.fixed_N > 0 ? get_params_vector(chain, "d_beta", M.fixed_N)' : zeros(M.fixed_N, N_samples)
    if M.fixed_N > 0
        for d in 1:M.fixed_N
            p_name = Symbol("d_beta[$d]")
            if p_name in Symbol.(name_strs)
                for s in 1:N_samples
                    fixed_effects_samples[d, s] = chain[p_name].data[s]
                end
            end
        end
    end

    
    for s in 1:N_samples
        s_sigma_sample = "s_sigma" in name_strs ? chain[:s_sigma].data[s] : 1.0
        local field_structured_sample = zeros(M.s_N)
        local field_noisy_sample = zeros(M.s_N)

        if M.model_space == "bym2"
            rho_sample = "s_rho" in name_strs ? clamp(chain[:s_rho].data[s], 0.0, 1.0) : 1.0
            icar_sample = vec(chain[:s_icar].data[s])
            iid_sample = vec(chain[:s_iid].data[s])
            field_structured_sample = s_sigma_sample .* sqrt(rho_sample) .* icar_sample
            field_noisy_sample = field_structured_sample .+ (s_sigma_sample .* sqrt(1.0 - rho_sample) .* iid_sample)
        elseif M.model_space in ["besag", "icar"]
            field_structured_sample = vec(chain[:s_icar].data[s]) .* s_sigma_sample
            field_noisy_sample = field_structured_sample

        elseif M.model_space == "leroux"
            rho_sample = "s_rho" in name_strs ? clamp(chain[:s_rho].data[s], 0.0, 1.0) : 1.0
            # Leroux uses a specific precision form, but for visualization we extract the latent field
            field_structured_sample = vec(chain[:s_raw].data[s]) .* s_sigma_sample
            field_noisy_sample = field_structured_sample
 
        elseif M.model_space == "sar"
            field_noisy_sample = vec(chain[:s_sar_raw].data[s]) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "bgcn"
            D_inv_sqrt = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
            W_norm = D_inv_sqrt * M.W * D_inv_sqrt
            field_noisy_sample = (W_norm * vec(chain[:gcn_weight_raw].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "fft"
            field_noisy_sample = (M.s_Q \ vec(chain[:s_spectral_raw].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "svgp"
            field_noisy_sample = (M.Z_inducing_proj * vec(chain[:u_latent].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "mosaic"
            field_noisy_sample = vec(chain[:mu_local].data[s])[M.cluster_assignments] .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "warping"
            w_warp = vec(chain[:w_warp].data[s])
            warp_proj = (M.s_coord * M.W_fixed) .+ M.b_fixed'
            field_noisy_sample = (sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "fitc"
            u_ind = vec(chain[:s_inducing].data[s])
            field_noisy_sample = (M.K_XZ * (M.K_ZZ \ u_ind))
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "fitcGP"
            field_noisy_sample = (M.Z_inducing_proj * vec(chain[:u_inducing].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
                
        elseif M.model_space =="denseGP"   
            field_noisy_sample = vec(chain[:s_eta_raw].data[s]) .* s_sigma_sample
            field_structured_sample = field_noisy_sample

        elseif M.model_space == "nystrom"
            field_noisy_sample = (M.K_nystrom_proj * vec(chain[:v_latent].data[s])) .* s_sigma_sample
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "deepGP"
            field_noisy_sample = vec(chain[:eta_gp].data[s])
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "dag"
            field_noisy_sample = vec(chain[:s_eta].data[s])
            field_structured_sample = field_noisy_sample
        elseif M.model_space == "local"
            mu_cl = vec(chain[:mu_clusters].data[s])
            field_structured_sample = mu_cl[M.cluster_assignments] .+ (vec(chain[:s_eta_raw].data[s]) .* s_sigma_sample)
            field_noisy_sample = field_structured_sample
        elseif M.model_space == "svc"
            field_structured_sample = vec(chain[:svc_raw].data[s]) .* s_sigma_sample
            field_noisy_sample = field_structured_sample
        elseif M.model_space == "iid"
            field_noisy_sample = vec(chain[:s_iid].data[s]) .* s_sigma_sample
            field_structured_sample = zeros(M.s_N)
        else
            field_noisy_sample = "s_eta" in name_strs ? vec(chain[:s_eta].data[s]) : zeros(M.s_N)
            field_structured_sample = field_noisy_sample
        end

        s_idx_obs = M.s_idx isa AbstractVector ? M.s_idx : fill(M.s_idx[1], M.y_N)
        s_eta_tot[1:M.y_N, s] = field_noisy_sample[s_idx_obs]
        if N_PS > 0  && !isnothing(PS)
            s_eta_tot[(M.y_N+1):end, s] = field_noisy_sample[PS.s_idx]
        end
        
        s_eta_obs_denoised[:, s] = field_structured_sample[s_idx_obs]
        s_eta_obs_noisy[:, s] = field_noisy_sample[s_idx_obs]

        if N_PS > 0 && !isnothing(PS)
            s_eta_tot[(M.y_N+1):end, s] = field_noisy_sample[PS.s_idx]
        end

        # 2. Temporal Manifolds
        t_sigma_sample = "t_sigma" in name_strs ? chain[:t_sigma].data[s] : 1.0
        if "t_eta" in name_strs;
            t_eta_tot[:, s] = vec(chain[:t_eta].data[s])
        elseif "t_raw" in name_strs; 
            t_eta_tot[:, s] = vec(chain[:t_raw].data[s]) .* t_sigma_sample
        elseif "t_alpha" in name_strs; 
            t_eta_tot[:, s] = (chain[:t_alpha].data[s] .* sin.(M.t_angle) .+ chain[:t_beta].data[s] .* cos.(M.t_angle)) .* t_sigma_sample
        end
       
        # 2. Temporal Manifolds
        if M.model_time == "logistic_ar1"
            if "m" in name_strs; m_latent_tot[:, s] = vec(chain[:m].data[s]); end
            t_eta_tot[:, s] = "t_eta" in name_strs ? vec(chain[:t_eta].data[s]) : vec(chain[:t_innovations].data[s])
        elseif M.model_time == "ar1"
            t_eta_tot[:, s] = vec(chain[:t_raw].data[s]) .* chain[:t_sigma].data[s]
        elseif M.model_time == "rw2"
            t_eta_tot[:, s] = vec(chain[:t_raw].data[s])
        elseif M.model_time == "harmonic"
            ta, tb = chain[:t_alpha].data[s], chain[:t_beta].data[s]
            t_eta_tot[:, s] = (ta .* sin.(M.t_angle) .+ tb .* cos.(M.t_angle)) .* chain[:t_sigma].data[s]
        end

        # 3. Seasonal Manifolds
        if M.model_season == "ar1"
            u_eta_tot[:, s] = vec(chain[:u_raw].data[s]) .* chain[:u_sigma].data[s]
        elseif M.model_season == "rw2"
            u_eta_tot[:, s] = vec(chain[:u_raw].data[s])
        elseif M.model_season == "harmonic"
            ua, ub = chain[:u_alpha].data[s], chain[:u_beta].data[s]
            u_eta_tot[:, s] = (ua .* sin.(M.u_angle) .+ ub .* cos.(M.u_angle)) .* chain[:u_sigma].data[s]
        end
  
        # 4. Space-Time Interaction Manifolds
        # Process Space-Time Interaction / Transport Field
        local current_st_field = zeros(M.s_N, M.t_N)
        if is_transport
            diff_val = Float64(st_diffusion_samples[s])
            adv_val = Float64(st_advection_samples[s])
            st_sigma_val = Float64(st_sigma_transport_samples[s])
            st_rho_persist_s = vec(Float64.(st_rho_persist_samples[s, :]))
            s_rho_for_advection = vec(Float64.(s_rho_samples[s, :])) # Use s_rho for advection base

            st_innov_sample = reshape(vec(Float64.(st_eta_z_samples[s, :])), M.s_N, M.t_N) .* st_sigma_val
            current_st_field[:, 1] = st_innov_sample[:, 1]

            for t in 2:M.t_N
                # PDE: Next = Persistence * (Prev - Diffusion - Advection) + Innovation
                mu_phys = current_st_field[:, t-1] .- diff_val .* (L * current_st_field[:, t-1]) .- adv_val .* (L * s_rho_for_advection)
                current_st_field[:, t] = logistic.(st_rho_persist_s) .* mu_phys .+ st_innov_sample[:, t]
            end
            st_eta_tot[:, :, s] = current_st_field
        elseif M.model_st in ["I", "II", "III", "IV"] 
            if "st_raw" in name_strs
                st_eta_tot[:, :, s] = reshape(vec(chain[:st_raw].data[s]) .* chain[:st_sigma].data[s], M.s_N, M.t_N)
            end
        end

    end



    eta = zeros(N_tot, N_samples)
    for s in 1:N_samples
        # st_eta = M.model_st == "none"  ? zeros(M.s_N, M.t_N) : st_eta_tot[:, :, s]
        for i in 1:N_tot
            is_obs = i <= M.y_N
            idx_adj = is_obs ? i : i - M.y_N
            # src = is_obs ? M : PS
            # a = src.s_idx isa AbstractVector ? src.s_idx[idx_adj] : src.s_idx
                        
            t = Int(is_obs ? (M.t_idx isa AbstractVector ? M.t_idx[i] : M.t_idx) : PS.t_idx[idx_adj])
            val = (is_obs ? M.log_offset[i] : 0.0) + s_eta_tot[i, s] + t_eta_tot[t, s]

            if M.model_season != "none"
                u = Int(is_obs ? (M.u_idx isa AbstractVector ? M.u_idx[i] : M.u_idx) : PS.u_idx[idx_adj])
                val += u_eta_tot[u, s]
            end

            # Apply fixed effects if defined in model
 
            if M.fixed_N > 0
                # Use hascolumn for DataFrames safety
                f_mat = if is_obs
                    M.fixed
                elseif !isnothing(PS) && hasproperty(PS, :surface_df) && hasproperty(PS.surface_df, :fixed)
                    PS.surface_df[!, :fixed]
                else
                    zeros(N_PS, M.fixed_N)
                end
                val += dot(fixed_effects_samples[:, s], f_mat[idx_adj, :])
            end

            # covariate effects
            if "c_eta" in name_strs
                c_level = Int(is_obs ? M.cov_indices[idx_adj, 1] : PS.z_indices[idx_adj, 1])
                val += c_eta_all[c_level, s]
            end
            eta[i, s] = val
        end
    end

    
    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)

    # Post-Stratification Weight Calculation
    s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N
    stratum_map = [(t_idx_obs[i] - 1) * s_N + s_idx_obs[i] for i in 1:M.y_N]
    obs_preds = denoised_mu[1:M.y_N, :]
    strata_preds = denoised_mu[M.y_N .+ stratum_map, :]
    weights = strata_preds ./ (obs_preds .+ 1e-9)

    return (
        spatial_structured = summarize_array(reshape(s_eta_obs_denoised, M.y_N, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eta_obs_noisy, M.y_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eta_tot, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(u_eta_tot, M.u_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = M.fixed_N > 0 ? summarize_array(reshape(fixed_effects_samples, M.fixed_N, 1, N_samples); alpha=alpha) : nothing,
        covariate_random_effects = summarize_array(reshape(c_eta_all, M.N_cat, 1, N_samples); alpha=alpha),
        predictions_observed = summarize_array(reshape(denoised_mu[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        predictions_noisy = N_PS > 0 ? summarize_array(reshape(noisy_y[(M.y_N+1):end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = (mean=vec(mean(weights, dims=2)), samples=weights),
        waic = _compute_waic(log_lik_mat[:, 1:M.y_N]), 
        family = fam, 
        arch = arch
    )
end



function _reconstruct(arch::ComplexArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    """
    _reconstruct(arch, fam, chain, M, alpha)

    Post-processing for models using spectral bases (RFF, FFT) or SPDE mesh approximations.
    
    Key Distinction:
    Unlike GMRF models where effects are area-indexed, spectral models typically provide
    realized latent fields (s_eff/f_gp) directly at the observation level (y_N).
    """
      
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    spatial = zeros(M.y_N, N_samples)
    temporal = zeros(M.t_N, N_samples)
    eta = zeros(M.y_N, N_samples)

    # 1. Realized Spatial Field Extraction
    target_field = "s_eff" in name_strs ? :s_eff : ("f_gp" in name_strs ? :f_gp : nothing)
    if !isnothing(target_field)
        for s in 1:N_samples; spatial[:, s] = vec(chain[target_field].data[s]); end
    end

    # 2. Temporal Trend and Interaction
    if "f_tm_raw" in name_strs
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        for s in 1:N_samples; temporal[:, s] = vec(chain[:f_tm_raw].data[s]) .* sig_tm[s]; end
    end

    st_int_present = "st_int_raw" in name_strs
    sig_int = st_int_present ? vec(chain[:sigma_int].data) : zeros(N_samples)

    # 3. Predictor Assembly (M.y_N level projection)
    beta_z = "beta_z" in name_strs ? vec(chain[:beta_z].data) : zeros(N_samples)
    beta_cov_eff = _extract_beta_cov(name_strs, chain, M, N_samples, alpha)

    for s in 1:N_samples
        st_i = st_int_present ? reshape(vec(chain[:st_int_raw].data[s]) .* sig_int[s], M.s_N, M.t_N) : zeros(M.s_N, M.t_N)
        for i in 1:M.y_N
            a, t = M.s_idx[i], M.t_idx[i]
            eta[i, s] = M.log_offset[i] + spatial[i, s] + temporal[t, s] + st_i[a, t] + beta_z[s] * M.z_obs[i]
            for k in 1:4
                p = Symbol("beta_cov[$k][$(M.cov_indices[i, k])]")
                if p in Symbol.(name_strs); eta[i, s] += chain[p].data[s]; end
            end
        end
    end

    actual_fam = (fam isa UnknownFamily) ? PoissonFamily() : fam
    denoised, noisy, log_lik = _process_ll_and_predictions(actual_fam, eta, chain, M, M.y_N, N_samples)

    return (spatial_structured = summarize_array(reshape(spatial, M.y_N, 1, N_samples); alpha=alpha),
            spatial_unstructured = nothing,
            spatial = summarize_array(reshape(spatial, M.y_N, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(temporal, M.t_N, 1, N_samples); alpha=alpha),
            seasonal = nothing,
            beta_cov = beta_cov_eff,
            predictions_denoised = summarize_array(reshape(denoised, M.y_N, 1, N_samples); alpha=alpha),
            predictions_noisy = summarize_array(reshape(noisy, M.y_N, 1, N_samples); alpha=alpha),
            waic = _compute_waic(log_lik),
            family = actual_fam, arch = arch)
end



function _reconstruct(arch::UnknownArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    """
    _reconstruct(arch, fam, chain, M, alpha)

    Post-processing for Deep Gaussian Processes, Mosaic Experts 

    Mathematical Logic:
    1. Manifold Extraction: Extracts the realized latent field from the GP manifold (f_latent, eta_gp, etc.).
    2. Temporal Trend: Supports additive AR1 trends if defined outside the GP kernel.
    3. Seasonal Effects: Reconstructs harmonic components (sin/cos) for periodic models.
    4. Prediction: Maps latent states through inverse-link functions with offset scaling.
    """
     
    N_samples = size(chain, 1)
    name_strs = string.(FlexiChains.parameters(chain))

    spatial = zeros(M.y_N, N_samples)
    temporal = zeros(M.t_N, N_samples)
    eta = zeros(M.y_N, N_samples)

    # 1. Latent Manifold Extraction
    target = "f_latent" in name_strs ? :f_latent : ("eta_gp" in name_strs ? :eta_gp : ("eta_spatial_time" in name_strs ? :eta_spatial_time : nothing))
    if !isnothing(target)
        for s in 1:N_samples; spatial[:, s] = vec(chain[target].data[s]); end
    end

    # 2. Temporal (if explicitly defined outside the GP manifold)
    if "f_tm_raw" in name_strs
        sig_tm = "sigma_tm" in name_strs ? vec(chain[:sigma_tm].data) : ones(N_samples)
        for s in 1:N_samples; temporal[:, s] = vec(chain[:f_tm_raw].data[s]) .* sig_tm[s]; end
    end

    # 3. Covariates and Seasonal Effects
    beta_cov_eff = _extract_beta_cov(name_strs, chain, M, N_samples, alpha)

    # Check for seasonal harmonic coefficients if present in Adaptive RFF models
    has_seasonal = "beta_cos" in name_strs && "beta_sin" in name_strs
    seasonal_post = zeros(M.t_N, N_samples)
    if has_seasonal
        bc, bs = vec(chain[:beta_cos].data), vec(chain[:beta_sin].data)
        for s in 1:N_samples
            t_vals = collect(1:M.t_N)
            seasonal_post[:, s] = bc[s] .* cos.(2pi .* t_vals ./ 12.0) .+ bs[s] .* sin.(2pi .* t_vals ./ 12.0)
        end
    end

    # 4. Assembly
    for s in 1:N_samples
        for i in 1:M.y_N
            t = M.t_idx[i]
            eta[i, s] = M.log_offset[i] + spatial[i, s] + temporal[t, s] + (has_seasonal ? seasonal_post[t, s] : 0.0)
            for k in 1:4
                p = Symbol("beta_cov[$k][$(M.cov_indices[i, k])]")
                if p in Symbol.(name_strs); eta[i, s] += chain[p].data[s]; end
            end
        end
    end

    actual_fam = (fam isa UnknownFamily) ? GaussianFamily() : fam
    denoised, noisy, log_lik = _process_ll_and_predictions(actual_fam, eta, chain, M, M.y_N, N_samples)

    return (spatial_structured = summarize_array(reshape(spatial, M.y_N, 1, N_samples); alpha=alpha),
            spatial_unstructured = nothing,
            spatial = summarize_array(reshape(spatial, M.y_N, 1, N_samples); alpha=alpha),
            temporal = summarize_array(reshape(temporal, M.t_N, 1, N_samples); alpha=alpha),
            seasonal = has_seasonal ? summarize_array(reshape(seasonal_post, M.t_N, 1, N_samples); alpha=alpha) : nothing,
            beta_cov = beta_cov_eff,
            predictions_denoised = summarize_array(reshape(denoised, M.y_N, 1, N_samples); alpha=alpha),
            predictions_noisy = summarize_array(reshape(noisy, M.y_N, 1, N_samples); alpha=alpha),
            waic = _compute_waic(log_lik),
            family = actual_fam, arch = arch)
end

 




function convert_advi_to_reconstruct_format(msol, model::DynamicPPL.Model, n_samples::Int=500)
    """
    Synopsis: Converts ADVI variational solutions into a format compatible with _reconstruct logic.
    Inputs:
    - msol: The solution object from Turing.vi()
    - model: The Turing model instance used for VI.
    - n_samples: Number of posterior samples to draw from the variational distribution.
    Outputs:
    - A FlexiChains-like object or standard Chain that _reconstruct can process.
    """
  
    # 1. Sample from the variational solution
    # Turing's rand(msol, n_samples) returns a Vector of VarNamedTuples
    samples_vec = rand(msol, n_samples)

    # 2. Extract unique base parameter names for reconstruction logic
    # We peek at the first sample to find the keys
    all_keys = keys(samples_vec[1])
    unique_bases = Set{Symbol}()
    for k in all_keys
        # Regex handles both scalar 'x' and vector 'x[1]' patterns
        m = match(r"^([^\\\\[]+)", string(k))
        if m !== nothing
            push!(unique_bases, Symbol(m.captures[1]))
        end
    end

    # 3. Create the reconstruct_samples (Vector of NamedTuples)
    # Required for DynamicPPL.reconstruct(model, sample)
    reconstruct_samples = map(samples_vec) do samp
        sample_params = Dict{Symbol, Any}()
        for base_sym in unique_bases
            base_str = string(base_sym)
            # Filter keys belonging to this base parameter group
            col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

            if length(col_keys) == 1 && string(first(col_keys)) == base_str
                sample_params[base_sym] = samp[first(col_keys)]
            else
                # Sort indexed keys like x[1], x[10], x[2] into numerical order
                sorted_keys = sort(collect(col_keys), by = k -> begin
                    m_idx = match(r"\\\\[(\\d+)\\\\]", string(k))
                    m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                end)
                sample_params[base_sym] = [samp[k] for k in sorted_keys]
            end
        end
        return (; sample_params...)
    end

    # 4. Construct FlexiChain for diagnostics
    # FIXED: Explicitly convert VarName keys to Symbol for FlexiChains compatibility
    formatted_dicts = map(samples_vec) do samp
        # Use Symbol(k) to ensure keys are Parameter{Symbol} not Parameter{VarName}
        Dict(FlexiChains.Parameter(Symbol(k)) => v for (k, v) in pairs(samp))
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples)
end


function convert_optim_to_reconstruct_format(optim_result, model, n_samples::Int=500; use_hessian=true, external_hessian=nothing)
    """
    Synopsis: Converts a point estimate (MAP/ML) from Optim/Optimisers into a distribution of samples.
    If a Hessian is available, it samples from the Multivariate Normal Laplace approximation.
    Otherwise, it creates a narrow Gaussian around the point estimate.
    """

    point_est_constrained = optim_result.params # This is the named tuple of constrained parameters

    reconstruct_samples_namedtuple = [] # Store NamedTuple for easier handling

    # Determine which Hessian to use
    H_to_use = nothing
    if external_hessian !== nothing
        H_to_use = external_hessian
    elseif hasproperty(optim_result, :hessian)
        H_to_use = optim_result.hessian
    end

    # Attempt Hessian-based sampling if enabled and a Hessian is available from either source
    if use_hessian && H_to_use !== nothing
        try
            # Get the unconstrained minimizer (mu_unconstrained)
            mu_unconstrained = optim_result.minimizer
            H = H_to_use

            # Ensure H is symmetric and compute its inverse for the covariance matrix
            Sigma = inv(Symmetric(Matrix(H) + Diagonal(fill(1e-6, size(H, 1)))))
            
            dist = MvNormal(mu_unconstrained, Sigma)

            # Generate n_samples of unconstrained parameters
            unconstrained_samples_matrix = rand(dist, n_samples)

            # Prepare template for conversion
            vi_template = DynamicPPL.VarInfo(model)

            for i in 1:n_samples
                sample_unconstrained_vec = unconstrained_samples_matrix[:, i]
                vi_current_sample = deepcopy(vi_template)
                DynamicPPL.setlink!(vi_current_sample, sample_unconstrained_vec)
                
                # Convert back to constrained named tuple
                constrained_sample_params = DynamicPPL.vi_to_params(vi_current_sample, model)
                push!(reconstruct_samples_namedtuple, constrained_sample_params)
            end

        catch e
            @warn "Failed to compute covariance from Hessian or convert samples, falling back to adding noise: $e"
            use_hessian = false 
        end
    end

    # Fallback to noise-based sampling
    if !use_hessian || isempty(reconstruct_samples_namedtuple)
        all_keys = keys(point_est_constrained)
        unique_bases = Set{Symbol}()
        for k in all_keys
            m = match(r"^([^\\[]+)", string(k))
            if m !== nothing
                push!(unique_bases, Symbol(m.captures[1]))
            end
        end

        reconstruct_samples_namedtuple = map(1:n_samples) do _
            sample_params_dict = Dict{Symbol, Any}() 
            for base_sym in unique_bases
                base_str = string(base_sym)
                col_keys = filter(k -> string(k) == base_str || startswith(string(k), "$base_str["), all_keys)

                if length(col_keys) == 1 && string(first(col_keys)) == base_str
                    val = point_est_constrained[first(col_keys)]
                    if val isa AbstractVector
                        sample_params_dict[base_sym] = val .+ randn() * 1e-4
                    else
                        sample_params_dict[base_sym] = val + randn() * 1e-4
                    end
                else
                    sorted_keys = sort(collect(col_keys), by = k -> begin
                        m_idx = match(r"\\[(\\d+)\\]", string(k))
                        m_idx !== nothing ? parse(Int, m_idx.captures[1]) : 0
                    end)
                    sample_params_dict[base_sym] = [point_est_constrained[k] + randn() * 1e-4 for k in sorted_keys]
                end
            end
            return (; sample_params_dict...)
        end
    end

    # 4. Format into a FlexiChain
    # IMPORTANT: Store ONLY the base symbols (e.g., :s_icar as a vector).
    # FlexiChains will automatically expand these into indexed names for summary statistics,
    # avoiding duplicate key errors while allowing _reconstruct to find the full vector.
    formatted_dicts = map(1:n_samples) do i
        samp = reconstruct_samples_namedtuple[i]
        d = Dict{FlexiChains.Parameter, Any}()
        for k in keys(samp)
            d[FlexiChains.Parameter(k)] = samp[k]
        end
        return d
    end

    chn = FlexiChains.FlexiChain{Symbol}(n_samples, 1, formatted_dicts)

    return (chain=chn, reconstruct_samples=reconstruct_samples_namedtuple)
end

struct bstm_Likelihood{F, Z, W, P, R, S, T, TR} <: ContinuousMultivariateDistribution
    family::F
    use_zi::Z
    weights::W
    phi_zi::P
    r_nb::R
    sigma_y::S
    trials::T
    y_obs::TR
end

Base.length(d::bstm_Likelihood) = length(d.y_obs)

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    
    total_lp = 0.0
    trials_scalar = d.trials isa Number    # size(d.trials,1) == 1 ? true : false
    sig_scalar = d.sigma_y isa Number # size(d.sigma_y, 1) == 1
     
    for i in 1:length(eta)
        y_val = d.y_obs[i]
        lin_pred = eta[i]
        lp = 0.0

        if d.family == "poisson"
            mu = exp(lin_pred)
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * exp(-mu) + 1e-10)
                else
                    lp = log(1 - d.phi_zi + 1e-10) + logpdf(Poisson(mu), y_val)
                end
            else
                lp = logpdf(Poisson(mu), y_val)
            end
        elseif d.family == "binomial"
            p = logistic(lin_pred)
            ntrials = trials_scalar ? d.trials : d.trials[i]
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * pdf(Binomial(ntrials, p), 0) + 1e-10)
                else
                    lp = log(1 - d.phi_zi + 1e-10) + logpdf(Binomial(ntrials, p), y_val)
                end
            else
                lp = logpdf(Binomial(ntrials, p), y_val)
            end
        elseif d.family == "negbin"
            mu = exp(lin_pred)
            prob = d.r_nb / (d.r_nb + mu)
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * pdf(NegativeBinomial(d.r_nb, prob), 0) + 1e-10)
                else
                    lp = log(1 - d.phi_zi + 1e-10) + logpdf(NegativeBinomial(d.r_nb, prob), y_val)
                end
            else
                lp = logpdf(NegativeBinomial(d.r_nb, prob), y_val)
            end
        elseif d.family == "gaussian"
            sig = sig_scalar ? d.sigma_y : d.sigma_y[i]
            lp = logpdf(Normal(lin_pred, sig), y_val)
        elseif d.family == "lognormal"
            sig = sig_scalar ? d.sigma_y : d.sigma_y[i]
            lp = logpdf(LogNormal(lin_pred, sig), y_val)
        end
        total_lp += lp * d.weights[i]
    end
    return total_lp
end




function generate_sim_data(s_N=10, t_N=5; rndseed=42)
    Random.seed!(rndseed)
    n_total = s_N * t_N
    unique_pts = [(rand() * 100, rand() * 100) for _ in 1:s_N]
    s_coord_tuple = repeat(unique_pts, t_N)
    s_coord = vcat([collect(t)' for t in s_coord_tuple]...)
    t_coord = rand(n_total) .* (t_N+1) .+ 2000
    weights = ones(n_total)
    trials  = ones(Int, n_total)
    u_N = 12
    period=u_N
    trend = 0.05 .* t_coord[:,1]
    seasonal = 1.0 .* cos.(2pi .* t_coord[:,1] ./ period)
    temporal_effect =  1.0 .* ( trend .+ seasonal )
    spatial_effect = 1.5 .* sin.(s_coord[:,1] .*  2pi) .* cos.(s_coord[:,2] .*  2pi)
    sigma_y = 0.2
    observation_error = sigma_y .* randn(n_total)
    Z = randn(n_total)
    W1_true = 0.5 .* sin.(t_coord[:,1] ./ 5.0) .+ 0.5 .* Z
    W2_true = 0.5 .* cos.(t_coord[:,1] ./ 5.0) .- 0.3 .* Z
    W3_true = 0.2 .* (t_coord[:,1] ./ n_total) .+ 0.1 .* Z
    sigma_w1, sigma_w2, sigma_w3 = 0.1, 0.2, 0.3
    W1_obs = W1_true .+ randn(n_total) .* sigma_w1
    W2_obs = W2_true .+ randn(n_total) .* sigma_w2
    W3_obs = W3_true .+ randn(n_total) .* sigma_w3
    cov_continuous = hcat(W1_obs, W2_obs, W3_obs)

    y_obs = 1.0 .+  spatial_effect + temporal_effect .+  observation_error .+ W1_obs .+ W2_obs .+ W3_obs
    y_binary = y_obs .> (mean(y_obs) + 0.5)
    y_counts = abs.(Int.(round.(y_obs))) * 100
    
    # fixed (covariate) .. user must make orthogonal for factors .. GLM.jl has a function I think
    fixed = [ones(n_total) ]   # column of ones is the intercept
     
    return (
        s_coord_tuple=s_coord_tuple, s_coord=s_coord, t_coord=t_coord, weights=weights, trials=trials, 
        s_N=s_N, t_N=t_N, u_N=u_N,
        y_obs=y_obs, y_binary=y_binary, y_counts=y_counts, fixed=fixed,
        z_obs=Z, w_obs=cov_continuous
    )
end
 


function build_bstm_ar1_template(n::Int; method="time", scale=true)
    if n < 1
        return (matrix = zeros(0, 0), scaling_factor = 1.0)
    elseif n == 1
        return (matrix = ones(1, 1), scaling_factor = 1.0)
    end

    Q = spzeros(Float64, n, n)
    m_str = lowercase(string(method))

    if m_str in ["season", "cyclic", "periodic"]
        # Periodic boundary conditions (Ring Graph)
        for i in 1:n
            Q[i, i] = 2.0
            Q[i, mod1(i - 1, n)] -= 1.0
            Q[i, mod1(i + 1, n)] -= 1.0
        end
    else
        # Standard Linear boundary conditions (Path Graph)
        Q[1, 1] = 1.0
        for i in 2:(n - 1); Q[i, i] = 2.0; end
        Q[n, n] = 1.0
        for i in 1:(n - 1)
            Q[i, i + 1] = -1.0
            Q[i + 1, i] = -1.0
        end
    end

    sf = 1.0
    if scale
        # Filter for non-zero eigenvalues (Null space rank 1 for connected graphs)
        e_vals = eigvals(Matrix(Q))
        eigs = filter(x -> x > 1e-7, e_vals)
        if !isempty(eigs)
            sf = exp(mean(log.(eigs)))
        end
    end

    return (matrix = Matrix(Symmetric(Q) ./ sf), scaling_factor = sf)
end


# Helper for AD-compatible precision recomposition
function build_bstm_ar1_precision(template_mat::AbstractMatrix, rho::Real; noise=1e-6)
    T = typeof(rho)
    n = size(template_mat, 1)
    # The AR1 precision is defined as (I + rho^2*I - rho*W) / (1-rho^2)
    # Our template represents (D - W). We transform it back:
    # Since template = (D - W)/sf, and D=2I (approx), we recompose:
    Q_rho = (1.0 / (1.0 - rho^2 + T(noise))) .* ( (1.0 + rho^2) .* I(n) .+ (rho) .* template_mat )
    return Symmetric(Q_rho + (T(noise) * I))
end



function build_bstm_rw2_template(n::Int; noise=1e-6, scale=true)
    # 1. Construct the Second-Order Difference Matrix (D)
    # For RW2, D is an (n-2) x n matrix
    D = spzeros(Float64, n - 2, n)
    for i in 1:(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end

    # 2. Form the precision matrix Q = D' * D
    Q = D' * D

    # 3. Geometric Mean Scaling
    # Ensures that the variance parameter represents the marginal standard deviation
    sf = 1.0
    if scale
        # Filter for non-zero eigenvalues (RW2 has a null space of rank 2)
        eigs = filter(x -> x > 1e-6, eigvals(Matrix(Q)))
        if !isempty(eigs)
            sf = exp(mean(log.(eigs)))
        end
    end

    return (matrix =Matrix(Symmetric( Q ./ sf)), scaling_factor = sf)
end

function build_bstm_rw2_precision(template_mat::AbstractMatrix, sigma::Real; noise=1e-6)
    # Recompose precision using the marginal variance sigma^2
    # Q_final = (1 / sigma^2) * Q_template
    T = eltype(sigma)
    Q_final = (1.0 / (sigma^2 + noise)) .* template_mat
    
    # Add jitter for numerical stability
    return Symmetric(Matrix(Q_final) + (noise * I))
end

function build_bstm_harmonic_template(n::Int; noise=1e-6, scale=true)
    # Construct a cyclic RW2 matrix for harmonic smoothing
    # This maintains the periodic constraint where the last node wraps to the first
    Q = spzeros(Float64, n, n)
    
    for i in 1:n
        # Difference operator: (x_{i-1} - 2x_i + x_{i+1})
        # Using mod1 for cyclic wrapping
        prev = mod1(i - 1, n)
        curr = i
        nxt  = mod1(i + 1, n)
        
        # Add the squared difference components to the precision matrix
        # This represents the structure of (D'D) where D is cyclic second-order differences
        Q[curr, curr] += 4.0
        Q[prev, prev] += 1.0
        Q[nxt, nxt]   += 1.0
        
        Q[curr, prev] -= 2.0
        Q[prev, curr] -= 2.0
        Q[curr, nxt]  -= 2.0
        Q[nxt, curr]  -= 2.0
        
        Q[prev, nxt]  += 1.0
        Q[nxt, prev]  += 1.0
    end

    # Geometric Mean Scaling
    sf = 1.0
    if scale
        eigs = filter(x -> x > 1e-6, eigvals(Matrix(Q)))
        if !isempty(eigs)
            sf = exp(mean(log.(eigs)))
        end
    end

    return (matrix = Matrix(Symmetric(Q ./ sf)), scaling_factor = sf)
end

 
function build_bstm_harmonic_precision(template_mat::AbstractMatrix, sigma::Real; noise=1e-6)
    # Note this is identical to RW2 ... as they are very similar. ..
    # Recompose precision using the marginal variance sigma^2
    # Q_final = (1 / sigma^2) * Q_template
    T = eltype(sigma)
    Q_final = (1.0 / (sigma^2 + noise)) .* template_mat
    
    # Add jitter for numerical stability
    return Symmetric(Matrix(Q_final) + (noise * I))
end


 
function precompute_nystrom_projection(spatial_coords, inducing_points, kernel_func; jitter=1e-6)

    # Example Usage:
    # kernel = Matern32Kernel() ∘ ScaleTransform(1.0)
    # modinputs_reference.K_nystrom_proj = precompute_nystrom_projection(areal_units.centroids, Z_inducing, kernel)

    # println("Precomputing Nystrom Projection Matrix...")
    
    # 1. K_mm: Kernel matrix between inducing points (M x M)
    K_mm = kernelmatrix(kernel_func, RowVecs(inducing_points))
    K_mm_stable = Symmetric(K_mm + jitter * I)
    
    # 2. K_nm: Kernel matrix between all spatial units and inducing points (N x M)
    K_nm = kernelmatrix(kernel_func, RowVecs(spatial_coords), RowVecs(inducing_points))
    
    # 3. Projection: K_nm * inv(K_mm)
    # We use the backslash operator for better numerical stability than direct inversion
    K_nystrom_proj = K_nm / K_mm_stable
    
    # println("Generated projection matrix of size: ", size(K_nystrom_proj))
    return K_nystrom_proj
end



# --- Optimized Householder PCA Helper Functions ---

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Initializes the Identity matrix to be transformed
    U = Matrix{T}(I, nU, nU)

    for k in 1:n_factors
        # Extract the k-th Householder vector
        vk = v_mat[:, k]
        norm_v = norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            
            # --- O(K * N^2) Optimization ---
            # Naive: U = (I - 2vv') * U  => O(N^3)
            # Optimized: U = U - 2v * (v' * U) => O(N^2)
            # We first compute the row vector (v' * U), then perform an outer product update.
            
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    # Return only the first n_factors columns as the orthonormal loadings matrix
    return U[:, 1:n_factors]
end

function householder_transform(v, nU, n_factors, ltri_indices, pca_sd, pdef_sd, noise)
    T = eltype(v)
    v_mat = zeros(T, nU, n_factors)
    v_mat[ltri_indices] .= v

    # Generate Orthonormal Loadings using optimized transformation
    U = householder_to_eigenvector(v_mat, nU, n_factors)

    # Reconstruct Covariance Components
    # W = Loadings * Scaled Eigenvalues
    W = U * Diagonal(pca_sd)

    # Kmat is the full covariance matrix: WW' + Residual_Variance
    # We add a small noise term for numerical stability
    Kmat = W * W' + (pdef_sd^2 + noise) * I(nU)

    return Kmat, pca_sd, U
end



function eigenvector_to_householder(U_in::AbstractMatrix{T}, n_factors) where {T}
# --- Optimal Vector Extraction (Orthonormal to Householder) ---

# eigenvector_to_householder(U, n_factors)
#
# Description:
#   Extracts the Householder reflector vectors (v) from an orthonormal loadings matrix U.
#   This allows for initializing the Bayesian model from a frequentist PCA result.
#
# Complexity: O(K * N^2)

    nU = size(U_in, 1)
    # We work on a copy to avoid modifying the input
    U = copy(U_in)
    
    # Storage for the lower-triangular part of the v_mat
    # Each column k corresponds to the k-th Householder vector
    v_mat = zeros(T, nU, n_factors)

    for k in 1:n_factors
        # 1. Target vector is the k-th column of the current transformation
        # For the identity, we want U[k,k] to be 1 and others 0
        x = U[k:end, k]
        
        # 2. Standard Householder Reflection Math
        # v = x + sign(x[1]) * ||x|| * e1
        norm_x = norm(x)
        vk = copy(x)
        
        sign_x1 = x[1] >= 0 ? one(T) : -one(T)
        vk[1] += sign_x1 * norm_x
        
        norm_vk = norm(vk)
        if norm_vk > 1e-9
            vk = vk ./ norm_vk
            
            # 3. Apply the reflection to the rest of the matrix (Rank-1 update)
            # U[k:end, k:end] = (I - 2vv') * U[k:end, k:end]
            # Using the O(N^2) update trick
            v_transpose_U = vk' * U[k:end, k:end]
            U[k:end, k:end] -= 2.0 .* vk * v_transpose_U
            
            # 4. Store the reflector
            v_mat[k:end, k] .= vk
        end
    end

    return v_mat
end


function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end

;;

function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
end



function build_ar1_precision_matrix(n, rho, sigma)
    Q = zeros(eltype(rho), n, n)
    if n > 0
        inv_sigma2 = 1.0 / (sigma^2 + 1e-6)
        Q[1, 1] = inv_sigma2
        for i in 2:n
            Q[i, i] = (1 + rho^2) * inv_sigma2
            Q[i-1, i] = -rho * inv_sigma2
            Q[i, i-1] = -rho * inv_sigma2
        end
        Q[n, n] = inv_sigma2
    end
    return Symmetric(Q)
end



"""
    build_rw2_precision_matrix(n::Int)

Constructs a Second-Order Random Walk (RW2) precision matrix for a binned covariate.

Assumptions:
- The covariate is discretized into `n` equidistant bins.
- The RW2 prior assumes second-order differences are independent Normal: (x_i - 2x_{i-1} + x_{i-2}) ~ N(0, sigma^2).
- The resulting matrix is singular (rank n-2) unless a small ridge (e.g., 1e-6*I) is added.
"""
function build_rw2_precision_matrix(n::Int)
    Q = zeros(n, n)
    for i in 1:n
        if i == 1 || i == n
            Q[i, i] = 1.0
        elseif i == 2 || i == n - 1
            Q[i, i] = 5.0
        else
            Q[i, i] = 6.0
        end
        if i < n
            Q[i, i+1] = Q[i+1, i] = (i == 1 || i == n-1) ? -2.0 : -4.0
        end
        if i < n - 1
            Q[i, i+2] = Q[i+2, i] = 1.0
        end
    end
    return Symmetric(Q)
end


function get_params_matrix_sizestructured(chain, base_name, dims)
    # Optimized for chn[:param].data[sample][row, col] access pattern
    n_rows, n_cols = dims
    N_samples = size(chain, 1)

    if Symbol(base_name) in names(chain, :parameters)
        res = zeros(Float64, n_rows, n_cols, N_samples)
        data_container = chain[Symbol(base_name)].data

        for s in 1:N_samples
            samp_mat = data_container[s]
            if size(samp_mat) == (n_rows, n_cols)
                res[:, :, s] = samp_mat
            elseif size(samp_mat) == (n_cols, n_rows)
                res[:, :, s] = samp_mat'
            end
        end
        return res
    end
    return nothing
end



 
function summarize_array(samples::AbstractArray; alpha=0.05)
    if isempty(samples) || all(isnan, samples)
        return (mean = [NaN], median = [NaN], lower = [NaN], upper = [NaN])
    end

    dims = size(samples)
    n_dims = length(dims)

    # Calculate statistics across the sample dimension
    post_mean = dropdims(mean(samples, dims=n_dims), dims=n_dims)
    post_median = dropdims(median(samples, dims=n_dims), dims=n_dims)
    low_bound = dropdims(mapslices(x -> quantile(x, alpha/2), samples, dims=n_dims), dims=n_dims)
    high_bound = dropdims(mapslices(x -> quantile(x, 1 - alpha/2), samples, dims=n_dims), dims=n_dims)

    # FIX: Ensure outputs are ALWAYS Vectors, even if they contain only one element.
    # This prevents MethodError: vec(::Float64) in downstream code.
    force_vector(x) = begin
        if x isa AbstractArray
            if ndims(x) == 0 # It's a 0-dimensional array (e.g., Array{Float64, 0})
                return [Float64(x[])] # Extract scalar and wrap in vector
            else
                return vec(collect(x)) # For higher-dimensional arrays, flatten to vector
            end
        else # It's already a scalar (e.g., Float64)
            return [Float64(x)] # Wrap scalar in vector
        end
    end

    return (
        mean = force_vector(post_mean),
        median = force_vector(post_median),
        lower = force_vector(low_bound),
        upper = force_vector(high_bound)
    )
end
 



function model_results_comprehensive(model, result, M, areal_units; PS="lazy", alpha=0.05, time_slice=1, n_samples=500, z_cat=nothing, w_cat=nothing, kwargs...)
    # 1. Prediction Surface Logic
    actual_PS = if PS == "lazy"
        s_N, t_N = M.s_N, M.t_N
        u_N = hasproperty(M, :u_N) ? M.u_N : 1
        n_obs = length(M.s_idx)
        obs_df = DataFrame(
            s_idx = M.s_idx isa AbstractVector ? M.s_idx : fill(M.s_idx, n_obs),
            t_idx = M.t_idx isa AbstractVector ? M.t_idx : fill(M.t_idx, n_obs)
        )
        if hasproperty(M, :u_idx) && !isnothing(M.u_idx)
            obs_df[!, :u_idx] = M.u_idx isa AbstractVector ? M.u_idx : fill(M.u_idx, n_obs)
        end
        basis_df = hasproperty(obs_df, :u_idx) ?
                   expand_grid(s_idx = 1:s_N, t_idx = 1:t_N, u_idx = 1:u_N) :
                   expand_grid(s_idx = 1:s_N, t_idx = 1:t_N)
        full_surface = create_prediction_surface(basis_df, obs_df, areal_units;
            lambda_s = get(kwargs, :lambda_s, 2.0),
            lambda_t = get(kwargs, :lambda_t, 1.0))
        (s_idx = Int.(full_surface.s_idx), t_idx = Int.(full_surface.t_idx),
         u_idx = hasproperty(full_surface, :u_idx) ? Int.(full_surface.u_idx) : nothing,
         surface_df = full_surface)
    else
        PS
    end

    # 2. Extract Chain
    chain = if result isa Turing.Variational.VIResult
        convert_advi_to_reconstruct_format(result, model, n_samples).chain
    elseif result isa Turing.Optimisation.ModeResult
        convert_optim_to_reconstruct_format(result, model, n_samples; use_hessian=true).chain
    else
        total_s = size(result, 1)
        if total_s > n_samples
            result[iter = (total_s - n_samples + 1):total_s]
        else
            result
        end
    end

    # 3. Reconstruct
    arch = get_architecture(M.model_arch)
    fam = get_model_family(M.model_family)
    pstats = _reconstruct(arch, fam, chain, M, actual_PS, alpha)

    # 4. Metrics
   
    y_obs_all = M.y_obs
    good_idx = findall(!ismissing, y_obs_all)
    y_obs = Float64.(y_obs_all[good_idx])
    y_pred = pstats.predictions_observed.mean[good_idx]

    rmse = sqrt(mean((y_obs .- y_pred).^2))
    r2 = length(unique(y_pred)) > 1 ? cor(y_obs, y_pred)^2 : 0.0
    waic_val = hasproperty(pstats, :waic) ? first(pstats.waic) : 0.0

    ss = MCMCChains.summarize(MCMCChains.Chains(chain))
    rh = filter(!isnan, Array(ss[:, stat=At(:rhat)]))
    mean_rhat = isempty(rh) ? 1.0 : mean(rh)
    es = filter(!isnan, Array(ss[:, stat=At(:ess_bulk)]))
    mean_ess = isempty(es) ? n_samples : mean(es)

    metrics = (rmse=rmse, r2=r2, waic=waic_val, rhat=mean_rhat, ess=mean_ess)

    showtuples(metrics)


    # 5. Core Diagnostics Plotting
    p_ppc = scatter(y_pred, y_obs, title="PPC", xlabel="Pred", ylabel="Obs", alpha=0.5, legend=false)

    p_tm = nothing
    if hasproperty(pstats, :temporal) && !all(isnan, pstats.temporal.mean)
        m_vals = pstats.temporal.mean
        p_tm = plot(m_vals, title="Temporal Trend", legend=false)
    end

    p_weight = nothing
    if hasproperty(pstats, :post_strat_weights)
        p_weight = histogram(vec(pstats.post_strat_weights.samples), title="Weight Posterior", bins=30, legend=false)
    end

    p_sp = nothing
    try
        p_sp = plot_posterior_results(pstats, M, areal_units; effect=:spatial_structured)
    catch; end

    plot_list = filter(!isnothing, [p_ppc, p_tm, p_weight, p_sp])
    display(plot(plot_list..., layout=(2, 2), size=(900, 700)))

    # 6. Specialized Extensions
    println("\n--- Specialized Model Components ---")
    
    if hasproperty(pstats, :covariate_effects) && !isempty(pstats.covariate_effects)
        println("Plotting Binned Covariate Effects...")
        plot_binned_covariates((pstats=pstats,))
    end

    if hasproperty(pstats, :seasonal) && !all(isnan, pstats.seasonal.mean)
        println("Plotting Seasonal Cycle...")
        plot_seasonal_cycle((pstats=pstats,))
    end

    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects)
        println("Plotting Fixed Effects...")
        fe = pstats.fixed_effects
        p_fe = bar(fe.mean, title="Fixed Effects Estimates", xlabel="Index", ylabel="Coefficient", legend=false)
        display(p_fe)
    end

    return (metrics=metrics, pstats=pstats, chain=chain)
end

function get_vec(obj, key)  
    # Helper for robust indexing
    val = hasproperty(obj, key) ? getproperty(obj, key) : nothing
    val isa AbstractVector ? val : [val]
end

 
function _reconstruct(arch::UnivariateArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_syms = FlexiChains.parameters(chain)
    p_names_str = string.(p_syms)

    # 1. Global and Design Components
    eta = zeros(N_tot, N_samples)
    alpha_vals = vec(get_params_vector(chain, "alpha", 1))

    # design matrix projection
    fixed_effects_samples = M.fixed_N > 0 ? get_params_vector(chain, "beta_fixed", M.fixed_N) : zeros(N_samples, M.fixed_N)

    # 2. Covariate Random Effects (Vectorized Mapping)
    cov_effects_total = zeros(N_tot, N_samples)
    binned_summaries = Dict()

    if haskey(M, :cov_groups) && !isempty(M.cov_groups)
        cov_names = collect(keys(M.cov_groups))
        # Handle multiple sig_covs or single sigma
        sig_covs = "sig_covs" in p_names_str ? get_params_vector(chain, "sig_covs", length(cov_names)) : ones(N_samples, length(cov_names))

        for (i, c_sym) in enumerate(cov_names)
            n_bins = size(M.cov_re_structures[c_sym], 1)
            raw_eff = get_params_vector(chain, "raw_eff_" * string(c_sym), n_bins)
            scaled_eff = raw_eff .* sig_covs[:, i]

            # Save binned effect for results summary
            binned_summaries[c_sym] = summarize_array(reshape(scaled_eff', n_bins, 1, N_samples); alpha=alpha)

            # Map to Obs
            idx_obs = M.cov_groups[c_sym]
            cov_effects_total[1:M.y_N, :] .+= scaled_eff[:, idx_obs]'

            # Map to Prediction Surface
            if N_PS > 0 && haskey(PS, :cov_groups) && haskey(PS.cov_groups, c_sym)
                idx_ps = PS.cov_groups[c_sym]
                cov_effects_total[M.y_N+1:end, :] .+= scaled_eff[:, idx_ps]'
            end
        end
    end

    # 3. Manifold Recovery
    s_sigma = vec(get_params_vector(chain, "s_sigma", 1))
    s_eff_struct = zeros(M.s_N, N_samples)
    s_eff_noisy = zeros(M.s_N, N_samples)
    t_sigma = vec(get_params_vector(chain, "t_sigma", 1))
    t_eff = zeros(M.t_N, N_samples)
    st_eff = zeros(M.s_N, M.t_N, N_samples)
    u_eff = :u_raw in p_syms ? reduce(hcat, chain[:u_raw].data) .* vec(get_params_vector(chain, "u_sigma", 1))' : zeros(get(M, :u_N, 1), N_samples)

    for j in 1:N_samples
        icar = vec(chain[:s_icar].data[j])
        if M.model_space == "bym2"
            rho = clamp(chain[:s_rho].data[j], 0.0, 1.0)
            iid = vec(chain[:s_iid].data[j])
            s_eff_struct[:, j] = s_sigma[j] .* sqrt(rho) .* icar
            s_eff_noisy[:, j] = s_eff_struct[:, j] .+ (s_sigma[j] .* sqrt(1-rho) .* iid)
        else
            s_eff_struct[:, j] = icar .* s_sigma[j]
            s_eff_noisy[:, j] = s_eff_struct[:, j]
        end
        
        t_raw_key = :t_raw in p_syms ? :t_raw : :t_main_raw
        t_eff[:, j] = vec(chain[t_raw_key].data[j]) .* t_sigma[j]

        if M.model_st == "IV"
            st_eff[:, :, j] = chain[:st_icar_raw].data[j] .* chain[:st_sigma].data[j]
        elseif M.model_st in ["diffusion", "advection", "advection_diffusion"]
            st_eff[:, :, j] = reshape(vec(chain[:st_innov].data[j]), M.s_N, M.t_N) .* chain[:st_sigma].data[j]
        end
    end

    # 4. Linear Predictor Construction
    for j in 1:N_samples
        st_mat = st_eff[:, :, j]
        for i in 1:N_tot
            is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
            s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])

            val = alpha_vals[j] + (is_obs ? get(M, :log_offset, zeros(M.y_N))[i] : 0.0)
            val += s_eff_noisy[s_idx, j] + t_eff[t_idx, j] + st_mat[s_idx, t_idx]
            if hasproperty(src, :u_idx); val += u_eff[src.u_idx[idx], j]; end
            val += cov_effects_total[i, j]
            if M.fixed_N > 0; val += dot(fixed_effects_samples[j, :], is_obs ? M.fixed[i, :] : PS.fixed[idx, :]); end
            eta[i, j] = val
        end
    end

    y_sigma_samples = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)

    # 5. Weights Calculation
    s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N
    stratum_map = [(t_idx_obs[i] - 1) * s_N + s_idx_obs[i] for i in 1:M.y_N]
    weights = denoised_mu[M.y_N .+ stratum_map, :] ./ (denoised_mu[1:M.y_N, :] .+ 1e-9)

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eff_noisy, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(u_eff, size(u_eff, 1), 1, N_samples); alpha=alpha),
        fixed_effects = M.fixed_N > 0 ? summarize_array(reshape(fixed_effects_samples', M.fixed_N, 1, N_samples); alpha=alpha) : nothing,
        binned_random_effects = binned_summaries,
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        post_strat_weights = summarize_array(reshape(weights, M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed = summarize_array(reshape(denoised_mu[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik_mat[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end


function _reconstruct(arch::SizeStructuredArchitecture, fam::ModelFamily, chain, M, PS, alpha)
    N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    N_tot = M.y_N + N_PS
    N_samples = size(chain, 1)
    p_syms = FlexiChains.parameters(chain)
    p_names_str = string.(p_syms)

    # 1. Global and Design Components
    eta = zeros(N_tot, N_samples)
    alpha_vals = vec(get_params_vector(chain, "alpha", 1))

    # design matrix projection
    fixed_effects_samples = M.fixed_N > 0 ? get_params_vector(chain, "beta_fixed", M.fixed_N) : zeros(N_samples, M.fixed_N)

    # 2. Covariate Random Effects (Vectorized Mapping)
    cov_effects_total = zeros(N_tot, N_samples)
    binned_summaries = Dict()
    
    if haskey(M, :cov_groups) && !isempty(M.cov_groups)
        cov_names = collect(keys(M.cov_groups))
        sig_covs = get_params_vector(chain, "sig_covs", length(cov_names))

        for (i, c_sym) in enumerate(cov_names)
            n_bins = size(M.cov_re_structures[c_sym], 1)
            raw_eff = get_params_vector(chain, "raw_eff", n_bins)
            scaled_eff = raw_eff .* sig_covs[:, i]
            
            # Save binned effect for results summary
            binned_summaries[c_sym] = summarize_array(reshape(scaled_eff', n_bins, 1, N_samples); alpha=alpha)

            # Map to Obs
            idx_obs = M.cov_groups[c_sym]
            cov_effects_total[1:M.y_N, :] .+= scaled_eff[:, idx_obs]'

            # Map to Prediction Surface
            if N_PS > 0 && haskey(PS, :cov_groups) && haskey(PS.cov_groups, c_sym)
                idx_ps = PS.cov_groups[c_sym]
                cov_effects_total[M.y_N+1:end, :] .+= scaled_eff[:, idx_ps]'
            end
        end
    end

    # 3. Manifold Recovery
    s_sigma = vec(get_params_vector(chain, "s_sigma", 1))
    s_eff_struct = zeros(M.s_N, N_samples)
    s_eff_noisy = zeros(M.s_N, N_samples)
    t_sigma = vec(get_params_vector(chain, "t_sigma_main", 1))
    t_eff = zeros(M.t_N, N_samples)
    st_eff = zeros(M.s_N, M.t_N, N_samples)
    u_eff = :u_raw in p_syms ? reduce(hcat, chain[:u_raw].data) .* vec(get_params_vector(chain, "u_sigma", 1))' : zeros(get(M, :u_N, 1), N_samples)

    for j in 1:N_samples
        icar = vec(chain[:s_icar].data[j])
        if M.model_space == "bym2"
            rho = clamp(chain[:s_rho].data[j], 0.0, 1.0)
            iid = vec(chain[:s_iid].data[j])
            s_eff_struct[:, j] = s_sigma[j] .* sqrt(rho) .* icar
            s_eff_noisy[:, j] = s_eff_struct[:, j] .+ (s_sigma[j] .* sqrt(1-rho) .* iid)
        else
            s_eff_struct[:, j] = icar .* s_sigma[j]
            s_eff_noisy[:, j] = s_eff_struct[:, j]
        end
        t_eff[:, j] = vec(chain[:t_main_raw].data[j]) .* t_sigma[j]

        if M.model_st == "IV"
            st_eff[:, :, j] = chain[:st_icar_raw].data[j] .* chain[:st_sigma].data[j]
        elseif M.model_st in ["diffusion", "advection", "advection_diffusion"]
            st_eff[:, :, j] = reshape(vec(chain[:st_innov].data[j]), M.s_N, M.t_N) .* chain[:st_sigma].data[j]
        end
    end

    # 4. Linear Predictor Construction
    for j in 1:N_samples
        st_mat = st_eff[:, :, j]
        for i in 1:N_tot
            is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
            s_idx = Int(src.s_idx[idx]); t_idx = Int(src.t_idx[idx])
            
            val = alpha_vals[j] + (is_obs ? get(M, :log_offset, zeros(M.y_N))[i] : 0.0)
            val += s_eff_noisy[s_idx, j] + t_eff[t_idx, j] + st_mat[s_idx, t_idx]
            if hasproperty(src, :u_idx); val += u_eff[src.u_idx[idx], j]; end
            val += cov_effects_total[i, j]
            if M.fixed_N > 0; val += dot(fixed_effects_samples[j, :], is_obs ? M.fixed[i, :] : PS.fixed[idx, :]); end
            eta[i, j] = val
        end
    end

    y_sigma_samples = _extract_volatility(chain, p_names_str, N_tot, N_samples)
    denoised_mu, noisy_y, log_lik_mat = _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, M.y_obs)

    # 5. Weights Calculation
    s_idx_obs = M.s_idx; t_idx_obs = M.t_idx; s_N = M.s_N
    stratum_map = [(t_idx_obs[i] - 1) * s_N + s_idx_obs[i] for i in 1:M.y_N]
    weights = denoised_mu[M.y_N .+ stratum_map, :] ./ (denoised_mu[1:M.y_N, :] .+ 1e-9)

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_noisy = summarize_array(reshape(s_eff_noisy, M.s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = summarize_array(reshape(u_eff, size(u_eff, 1), 1, N_samples); alpha=alpha),
        fixed_effects = M.fixed_N > 0 ? summarize_array(reshape(fixed_effects_samples', M.fixed_N, 1, N_samples); alpha=alpha) : nothing,
        binned_random_effects = binned_summaries,
        volatility = summarize_array(reshape(y_sigma_samples, N_tot, 1, N_samples); alpha=alpha),
        post_strat_weights = summarize_array(reshape(weights, M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed = summarize_array(reshape(denoised_mu[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_denoised = N_PS > 0 ? summarize_array(reshape(denoised_mu[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        waic = _compute_waic(log_lik_mat[:, 1:M.y_N]),
        family = fam, arch = arch
    )
end


function plot_binned_covariates(res)
    # Check if covariate effects exist in pstats
    cov_effects = get(res.pstats, :covariate_effects, Dict())
    if isempty(cov_effects)
        println("No covariate effects found to plot.")
        return nothing
    end

    plts = []
    for (key, samples) in cov_effects
        # Key format: :raw_rw2_cwd
        name = replace(string(key), "raw_rw2_" => "")
        m = vec(mean(samples, dims=2))
        l = vec(quantile.(eachrow(samples), 0.025))
        u = vec(quantile.(eachrow(samples), 0.975))
        
        p = plot(m, ribbon=(m .- l, u .- m), title="Effect: $name", 
                 xlabel="Bin", ylabel="Value", legend=false, fillalpha=0.2)
        push!(plts, p)
    end
    
    n = length(plts)
    if n > 0
        display(plot(plts..., layout=(ceil(Int, n/2), 2), size=(900, 300*ceil(Int, n/2))))
    end
end

function plot_seasonal_cycle(res)
    if !hasproperty(res.pstats, :seasonal) || all(isnan, res.pstats.seasonal.mean)
        println("No seasonal component found to plot.")
        return nothing
    end
    
    m = res.pstats.seasonal.mean
    s = res.pstats.seasonal.samples
    l = vec(quantile.(eachrow(s), 0.025))
    u = vec(quantile.(eachrow(s), 0.975))
    
    plt = plot(m, ribbon=(m .- l, u .- m), title="Seasonal Cycle (Binned/Harmonic)", 
               xlabel="Season Bin", ylabel="Effect", lw=2, fillalpha=0.3, legend=false)
    display(plt)
    return plt
end


function dataframe_to_named_array(df::DataFrame)
    v_names = Symbol.(names(df))
    data_mat = Matrix(df)
    return NamedArray(data_mat, (1:size(data_mat, 1), v_names), ("Observation", "Variable"))
end


function get_optimal_sampler(model::DynamicPPL.Model;
    nuts_n_samples_adaptation=50,
    nuts_target_acceptance_ratio=0.65,
    pg_particles=20,
    kwargs...)

    # Generate initial samples to inspect parameter types
    init_samples = [Dict(pairs(rand(model))) for _ in 1:3]
    full_init_dict = init_samples[1]

    # 1. Identify Parameter Roles
    fixed_params = filter(k -> all(s -> s[k] == full_init_dict[k], init_samples), keys(full_init_dict))
    discrete_params = filter(k -> k ∉ fixed_params && (full_init_dict[k] isa Integer || full_init_dict[k] isa Bool), keys(full_init_dict))

    # 2. Gaussian Filter for ESS
    # ESS requires strictly Normal/MvNormal priors. 
    # We route AbstractVectors (latent fields) here, but exclude structured scales like sig_covs
    latent_fields = filter(k -> begin
        val = full_init_dict[k]
        k_str = string(k)
        k ∉ fixed_params && k ∉ discrete_params && val isa AbstractVector &&
        !occursin("sig_", k_str) && !occursin("rho", k_str) && !occursin("phi", k_str)
    end, keys(full_init_dict))

    # 3. NUTS for non-Gaussian/Positive parameters
    active_hypers = filter(k -> begin
        k ∉ fixed_params && k ∉ discrete_params && k ∉ latent_fields
    end, keys(full_init_dict))

    sampler_blocks = []
    if !isempty(discrete_params); push!(sampler_blocks, Tuple(discrete_params) => PG(pg_particles)); end
    if !isempty(latent_fields); push!(sampler_blocks, Tuple(latent_fields) => ESS()); end
    if !isempty(active_hypers); push!(sampler_blocks, Tuple(active_hypers) => Turing.NUTS(nuts_n_samples_adaptation, nuts_target_acceptance_ratio)); end
    if !isempty(fixed_params); push!(sampler_blocks, Tuple(fixed_params) => MH()); end

    inits = parameter_inits(model)
    return Gibbs(sampler_blocks...), inits
end

;;


