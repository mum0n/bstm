
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
  


 

Turing.@model function turing_car(D, W, X, log_offset, y, auid )
    # base model .. slow, MVN of var-covariance matrix .. didactic version
    nX=size(X,2)
    A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
    tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
    beta ~ filldist( Normal(0.0, 1.0), nX );
    prec = tau .* (D - A .* W)
    if !isposdef(prec)
        # check postive definiteness
        # phi ~ MvNormal( zeros(N) ); 
        Turing.@addlogprob! -Inf
        return nothing
    end
    sigma = inv( Symmetric(prec) )
    # sigma = Symmetric( prec) \ Diagonal(ones(nX))  # alternatively
    phi ~ MvNormal( sigma );  # mean zero
    mu =  X * beta .+ phi[auid] .+ log_offset 
    @. y ~ LogPoisson( mu );
end

   
Turing.@model function turing_car_prec(D, W, X, log_offset, y,  auid  )
    # MVN of precision matrix .. slightly faster
    nX=size(X,2)
    A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
    tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
    beta ~ filldist( Normal(0.0, 1.0), nX);
    prec = tau .* (D - A .* W)
    if !isposdef(prec)
        # check postive definiteness
        # phi ~ MvNormal( zeros(N) ); 
        Turing.@addlogprob! -Inf
        return nothing
    end
    phi ~ MvNormalCanon( Symmetric(prec) );  # mean zero .. no inverse
    mu =  X * beta .+ phi[auid] .+ log_offset 
    @. y ~ LogPoisson( mu );
end

 

Turing.@model function turing_icar_test( node1, node2; ysd=std(skipmissing(y)), nData=size(y,1)  )
    # equivalent to Morris' "simple_iar' .. testing pairwise difference formulation
    # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)

    phi ~ filldist( Normal(0.0, ysd), nData)   # 10 is std from data: std(y)=7.9 stan goes from U(-Inf,Inf) .. not sure why 
    dphi = phi[node1] - phi[node2]
    lp_phi =  -0.5 * dot( dphi, dphi )
    Turing.@addlogprob! lp_phi
    
    # soft sum-to-zero constraint on phi)
    # equivalent to mean(phi) ~ normal(0,0.001)
    sum_phi = sum(phi)
    sum_phi ~ Normal(0, 0.001 * nData);  
  
    # no data likelihood -- just prior sampling  -- 
end

  
Turing.@model function turing_icar_bym( X, log_offset, y, nX, node1, node2; ysd=std(skipmissing(y)),  nData=size(X,1), auid=1:nData )
    # BYM
    # A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
     # tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
     beta ~ filldist( Normal(0.0, 5.0), nX);
     theta ~ filldist( Normal(0.0, 1.0), nData) # unstructured (heterogeneous effect)
     # phi ~ filldist( Laplace(0.0, ysd), nData) # spatial effects: stan goes from -Inf to Inf .. 
     phi ~ filldist( Normal(0.0, ysd), nData) # spatial effects: stan goes from -Inf to Inf .. 
 
     # pairwise difference formulation ::  prior on phi on the unit scale with sd = 1
     # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
     dphi = phi[node1] - phi[node2]
     lp_phi =  -0.5 * dot( dphi, dphi )
     Turing.@addlogprob! lp_phi
     
     # soft sum-to-zero constraint on phi)
     # equivalent to mean(phi) ~ normal(0, 0.001)
     sum_phi = sum(phi)
     sum_phi ~ Normal(0, 0.001 * nData);  

     tau_theta ~ Gamma(3.2761, 1.0/1.81);  # Carlin WinBUGS priors
     tau_phi ~ Gamma(1.0, 1.0);            # Carlin WinBUGS priors

     sigma_theta = inv(sqrt(tau_theta));  # convert precision to sigma
     sigma_phi = inv(sqrt(tau_phi));      # convert precision to sigma

     mu =  X * beta .+ phi[auid] .* sigma_phi .+ theta .* sigma_theta .+ log_offset 
  
     @. y ~ LogPoisson( mu );
end
 

Turing.@model function turing_icar_bym2( X, log_offset, y, auid, nX, nAU, node1, node2, scaling_factor )
    beta ~ filldist( Normal(0.0, 1.0), nX);
    theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
    # pairwise difference formulation ::  prior on phi on the unit scale with sd = 1
    # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
    dphi = phi[node1] - phi[node2]
    Turing.@addlogprob! -0.5 * dot( dphi, dphi )
    # soft sum-to-zero constraint on phi)
    sum_phi = sum(phi)
    sum_phi ~ Normal(0, 0.001 * nAU);  
    sigma ~ truncated( Normal(0, 1.0), 0, Inf) ; 
    rho ~ Beta(0.5, 0.5);
    # variance of each component should be approximately equal to 1
    convolved_re =  sigma .*  ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi );
    mu =  X * beta +  convolved_re[auid] + log_offset 
    @. y ~ LogPoisson( mu );
end
 

Turing.@model function turing_icar_bym2_binomial(y, Ntrial, X, nX, nAU, node1, node2, scaling_factor)
    # poor form to not pass args directly but simpler to use global vars as this is a one-off:
    beta0 ~ Normal(0.0, 1.0)
    betas ~ filldist( Normal(0.0, 1.0), nX); #coeff
    theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
    dphi = phi[node1] - phi[node2]
    Turing.@addlogprob! (-0.5 * dot( dphi, dphi )) # directly add to logprob
    sum_phi = sum(phi)
    sum_phi ~ Normal(0.0, 0.001 * nAU);      # soft sum-to-zero constraint on phi), equivalent to 
    sigma ~ Gamma(1.0, 1.0)
    rho ~ Beta(0.5, 0.5);
    # variance of each component should be approximately equal to 1
    convolved_re =  sqrt(1 - rho) .* theta .+ sqrt(rho / scaling_factor) .* phi ;
    mu = beta0 .+ X * betas .+ sigma .* convolved_re 
    # y ~ arraydist(LazyArray(@~ BinomialLogit.(Ntrial, v)))  # 100 sec
    @. y ~ BinomialLogit(Ntrial, mu)
end



Turing.@model function turing_icar_bym2_groups( X, log_offset, y, auid, nX, nAU, node1, node2, scaling_factor, groups, gi; ysd=std(skipmissing(y)) )

    ## incomplete?


    # BYM2
    # A ~ Uniform(0.0, 1.0); # A = 0.9 ; A==1 for BYM / iCAR
     # tau ~ Gamma(2.0, 1.0/2.0);  # tau=0.9
     beta ~ filldist( Normal(0.0, 5.0), nX);
     theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
     phi ~ filldist(Normal(0.0, ysd), nAU) # spatial effects: stan goes from -Inf to Inf .. 
        
     # pairwise difference formulation ::  prior on phi on the unit scale with sd = 1
     # see (https://mc-stan.org/users/documentation/case-studies/icar_stan.html)
     dphi = phi[node1] - phi[node2]
     lp_phi =  -0.5 * dot( dphi, dphi )
     Turing.@addlogprob! lp_phi
     
     sigma ~ truncated( Normal(0, 1.0), 0, Inf) ; 
     rho ~ Beta(0.5, 0.5);

     convolved_re = zeros(nAU)

     # Threads.@threads add once working
     for j in 1:length(gi)
         ic = gi[j] 
        
         # soft sum-to-zero constraint on phi)
         # equivalent to mean(phi) ~ normal(0, 0.001)
         sum_phi = sum(phi[ic])
         sum_phi ~ Normal(0, 0.001 * nAU);  

         if  length(ic) == 1 
             convolved_re[ ic ] = sigma .* theta[ ic ];
         else  
             convolved_re[ ic ] = sigma .* ( sqrt.(1 .- rho) .* theta[ ic ]  +  sqrt(rho ./ scaling_factor[j] )  .* phi[ ic ] ) ;
         end 
     end
  
     # convolved_re =  sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi;
   
     mu =   X * beta .+  convolved_re[auid] .+ log_offset 
   
     @. y ~ LogPoisson( mu );
  
    # to compute from posteriors
    #  real logit_rho = log(rho / (1.0 - rho));
    #  vector[N] eta = log_E + beta0 + x * betas + convolved_re * sigma; // co-variates
    #  vector[N] mu = exp(eta);
end 

 

@memoize function icar_form(theta, phi, sigma, rho)
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    # Reibler parameterization: https://pubmed.ncbi.nlm.nih.gov/27566770/
    # https://www.jstatsoft.org/index.php/jss/article/view/v063c01/841
    sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )  
end
   

@memoize function sample_gaussian_process( ; GPmethod="cholesky", returntype="default",
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

            Yobs_mean_process = Kio' * ( Li' \ (Li \ Yinducing_mean_process ))  # mean process from inducing pts
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
            Yobs_mean_process = Ko' * ( Lo' \ (Lo \ Yobs ))  # mean process from inducing pts
            
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
 
 

Turing.@model function turing_glm_icar( ; family="poisson", GPmethod="GPsparse", 
    Y=nothing, YG=nothing, good=nothing, 
    X=nothing, G=nothing, Gp=nothing, nInducing=nothing, log_offset=nothing, 
    auid=nothing, nAU=nothing, node1=nothing, node2=nothing, scaling_factor=nothing, tuid=nothing,
    kerneltype="squared_exponential" )

    # almost a full random effect GLM (poisson, binomial and gaussian)
    # spatial random effects with Reibler parameterization
    # covariates (fixed and GP)
    # use this as a basis and strip out uneeded parts to optimize
 
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

    nData=length(Y)
    
    Ymu = zeros( nData ) 
    
    if !isnothing(log_offset)
        Ymu .+= log_offset 
    end

    if !isnothing(X)
        # fixed effects
        nX=size(X,2)
        beta ~ filldist( Normal(0.0, 1.0), nX )
        Ymu += X * beta
    end
 
    if !isnothing(G)
        # gaussian process for covariates G
        nG = size(G, 2)
          
        kernel_var ~ filldist( Exponential(1.0), nG )  # can't be much larger than 1 (already scaled)
        kernel_scale ~ filldist( LogNormal(1.0, 2.0), nG )  

        l2reg ~  filldist(Gamma(1.0, 0.001), nG )    # L2 regularization factor for ridge regression
        
#        Gymu = zeros(nInducing, nG)   # component-specific random effect

        sum_Gy_sample = zeros(nG)
        
        YGcurr = YG - Ymu

        for i in 1:nG

            ys = sample_gaussian_process( GPmethod=GPmethod, returntype="sample_loglik", 
                kerneltype=kerneltype,
                kvar=kernel_var[i], kscale=kernel_scale[i],
                Yobs=YGcurr, Xobs=G[:,i], Xinducing=Gp[:,i], lambda=l2reg[i]
            )
            
            Turing.@addlogprob! -ys.loglik 
            Ymu += ys.Yobs_sample
            # Gymu[:,i] ~  rand(fposterior(Gp[:,i], l2reg[i]))   # a mechanism to store sampled mean process 
               
            sum_Gy_sample[i] = sum(ys.Yobs_sample)  
            sum_Gy_sample[i] ~ Normal(0.0, 0.0001 )   # soft sum-to-zero constraint
        end

    end


    if !isnothing(auid)
 
        # spatial effects (without inverting covariance)  
        if isnothing(nAU)
            nAU = length(auid)
        end

        theta ~ filldist( Normal(0.0, 1.0), nAU )  # unstructured (heterogeneous effect)
        phi ~ filldist( Normal(0.0, 1.0), nAU ) # spatial effects: stan goes from -Inf to Inf .. 
    
        sigma ~ Exponential(1.0)   # == Gamma(1,1)
        rho ~   Beta(0.5, 0.5) 
        
        dphi = phi[node1] - phi[node2]
        dot_phi = dot( dphi, dphi )
        Turing.@addlogprob! -0.5 * dot_phi

        # soft sum-to-zero constraint on phi
        sum_phi = sum( dot_phi ) 
        sum_phi ~ Normal(0.0, 0.01 * nAU);      # soft sum-to-zero constraint on s_phi)
        
        Ymu += icar_form( theta, phi, sigma, rho )[auid] 

    end


    # ------------
    # data likelihood
    # 
    # notes: 
    # a method to truncate safely if needed: Ymu = max.(zero(eltype(Ymu)), Ymu) #
    # equivalent ways of expressing likelihood:
    # @. y ~ LogPoisson( Ymu);
    # y ~ arraydist([LogPoisson( Ymu[i] ) for i in 1:nData ])
    # y ~ arraydist(LazyArray(Base.broadcasted((l) -> LogPoisson(l), Ymu)))
    # y ~ arraydist(LazyArray( @~ LogPoisson.(Ymu) ) )

    if family=="poisson"
        Y ~ arraydist( @. LogPoisson( Ymu ) )   
    elseif family=="bernoulli"
        Y ~ arraydist( @. Bernoulli( logistic.(Ymu) ) ) 
    elseif family=="gaussian"
        Ysd ~ Exponential(1.0)
        Y ~ arraydist( @. Normal.( Ymu, Ysd ) ) 
    end

    return nothing
      
end
 

Turing.@model function turing_glm_icar_optimized( ; family="poisson", GPmethod="cholesky", 
    Y=nothing,  YG=nothing, good=nothing, 
    X=nothing, G=nothing, Gp=nothing, nInducing=nothing, log_offset=nothing, 
    auid=nothing, nAU=nothing, node1=nothing, node2=nothing, scaling_factor=nothing, tuid=nothing,
    kerneltype="squared_exponential", gpc=GPC()
    )

    # fast version  .. optimized as much as possible
    # almost a full random effect GLM (poisson, binomial and gaussian)
    # spatial random effects with Reibler parameterization
    # covariates (fixed and GP)
    # use this as a basis and strip out uneeded parts to optimize
 
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

    nData=length(Y)
   
    Ymu = zeros( nData ) 

    if !isnothing(log_offset)
        Ymu .+= log_offset 
    end

    # fixed and linear effects
    nX = size(X,2)
    # fixed effects
    beta ~ filldist( Normal(0.0, 1.0), nX )
    Ymu += X * beta

    # must have gaussian process for covariates G
    nG = size(G, 2)
    
    kernel_var ~ filldist( Exponential(1.0), nG )  # can't be much larger than 1 (already scaled)
    kernel_scale ~ filldist( LogNormal(1.0, 2.0), nG )  

    l2reg ~  filldist(Gamma(1.0, 0.01), nG )    # L2 regularization factor for ridge regression
    # l2reg = fill(0.001, nG)
#    Gymu = zeros(nInducing, nG)   # component-specific random effects
    sum_Gy_sample = zeros(nG) 
    # α = MvLogNormal(MvNormal(Zeros(nG), I))
    YGcurr = YG - Ymu

    for i in 1:nG
          
        if kerneltype=="default" || kerneltype=="squared_exponential"
            fkernal = kernel_var[i] * SqExponentialKernel() ∘ ScaleTransform( kernel_scale[i]) # ∘ ARDTransform(α)
        end

        if kerneltype=="matern32"
            fkernal = kernel_var[i] * Matern32Kernel() ∘ ScaleTransform( kernel_scale[i]) # ∘ ARDTransform(α)
        end
 
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)

        fobs = fgp( G[:,i], l2reg[i] )
        finducing = fgp( Gp[:,i], l2reg[i] ) 
        
        # only GPsparse or GPvfe 
        
        if !isnothing(match(r"GPsparse", GPmethod))
            fsparse = SparseFiniteGP(fobs, finducing)
            Turing.@addlogprob!  -AbstractGPs.logpdf(fsparse, YGcurr)
            fposterior = posterior(fsparse, YGcurr)
        end
        
        if !isnothing(match(r"GPvfe", GPmethod))
            fsparse = VFE( finducing ) 
            Turing.@addlogprob! -AbstractGPs.elbo( fsparse, fobs, YGcurr )                  
            fposterior = posterior( fsparse, fobs, YGcurr)  
        end

        Gy_sample = rand( fposterior( G[:,i],  l2reg[i] )  )
        sum_Gy_sample[i] = sum( Gy_sample ) 
        sum_Gy_sample[i] ~ Normal(0.0, 0.0001 * nData)  
        
        Ymu += Gy_sample

#        Gymu[:,i] ~  fposterior(Gp[:,i], l2reg[i]) # a mechanism to store sampled mean process 
    end
    
    # spatial effects (without inverting covariance)  
    theta ~ filldist( Normal(0.0, 1.0), nAU )  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU ) # spatial effects: stan goes from -Inf to Inf .. 

    sigma ~ Exponential(1.0)   
    rho ~   Beta(0.5, 0.5) 
    
    dphi = phi[node1] - phi[node2]
    dot_phi = dot( dphi, dphi )
    Turing.@addlogprob! -0.5 * dot_phi

    # soft sum-to-zero constraint on phi
    sum_phi = sum( dot_phi ) 
    sum_phi ~ Normal(0.0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)
    
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    Ymu += (sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi ))[auid] 

  
    # ------------
    # data likelihood
    # 
    # notes: 
    # a method to truncate safely if needed: Ymu = max.(zero(eltype(Ymu)), Ymu) #
    # equivalent ways of expressing likelihood:
    # @. y ~ LogPoisson( Ymu);
    # y ~ arraydist([LogPoisson( Ymu[i] ) for i in 1:nData ])
    # y ~ arraydist(LazyArray(Base.broadcasted((l) -> LogPoisson(l), Ymu)))
    # y ~ arraydist(LazyArray( @~ LogPoisson.(Ymu) ) )

    if family=="poisson"
        Y ~ arraydist( @. LogPoisson( Ymu ) )   
    elseif family=="bernoulli"
        Y ~ arraydist( @. Bernoulli( logistic.(Ymu) ) ) 
    elseif family=="gaussian"
        Ysd ~ Exponential(1.0)
        Y ~ arraydist( @. Normal.( Ymu, Ysd ) ) 
    end

    return nothing
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
    
    # @infiltrate

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
            offset = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$offset]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density",
                    margin=1.5mm,
                )
                push!(ps, p)
                offset += 1
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


@memoize function fkernal( kernfunctype="squared_exp", params=nothing )

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


@memoize sekernel2(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

sekernel(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

Turing.@model function test_gp0( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    
    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel(kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg 
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) 
          
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        
        fposterior = posterior( SparseFiniteGP(fobs, finducing), Yobs)
        
        Gy_sample = rand(fposterior(Xobs, lambda)) 
        Gymu_sample = rand(fposterior(Xinducing, lambda))
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        
        # Main.DEBUG = fposterior(Xinducing, lambda)

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
end
 


Turing.@model function test_gp1( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    sigma ~ LogNormal(0.0,1.0)

    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel( kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg + sigma^2
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) # implicit zero-mean
        
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        
        # fposterior = posterior( SparseFiniteGP(fobs, finducing), Yobs)
        
        # Distribution is MvNormal  sample from dense cov matrix
        Gy_sample ~ fgp( Xobs, lambda) 
        Gymu_sample ~ fgp(Xinducing, lambda)
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        
        # Main.DEBUG = fposterior(Xinducing, lambda)

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
    
end
 

Turing.@model function test_gp1sparse( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    sigma ~ LogNormal(0.0,1.0)

    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)
     
    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel( kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg + sigma^2
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) # implicit zero-mean
        
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        
        fposterior = posterior(SparseFiniteGP(fobs, finducing), Yobs)
        
        Main.DEBUG = fposterior

        # Distribution is MvNormal  sample from sparse cov matrix
        Gy_sample ~ fposterior( Xobs, lambda) 
        Gymu_sample ~ fposterior(Xinducing, lambda)
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
    
end
 
Turing.@model function test_gp1vfe( Y, YG, G, Gp, nInducing, good, i, gpc=GPC() )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    sigma ~ LogNormal(0.0,1.0)

    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel( kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg + sigma^2
       
        fgp = atomic(AbstractGPs.GP(fkernal), gpc) # implicit zero-mean
        
        fobs = fgp( Xobs, lambda )

        vfe = VFE( fgp(Xinducing, lambda ) ) 
        fposterior = posterior(vfe, fobs, Yobs)  # Distribution is MvNormal  
 
        Main.DEBUG = fposterior

        # Distribution is MvNormal  sample from sparse cov matrix
        Gy_sample ~ fposterior( Xobs, lambda) 
        Gymu_sample ~ fposterior(Xinducing, lambda)
                
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
   
    
end
 


Turing.@model function test_gp2( Y, YG, G, Gp, nInducing, good, i )
               
    Y = Y[good] 
    YG = YG[good]  
    G = G[good,:] 

    nData=length(Y)
    
    Ymu = zeros( nData )  

    nG = 1 # size(G, 2)
    i = 1
    kernel_var ~ LogNormal(0.0, 1.0)  
    kernel_scale ~  LogNormal(0.0, 1.0)   
    l2reg = 0.001    # L2 regularization factor for ridge regression
    
    Gymu = zeros(nInducing)   # component-specific random effect

    sum_Gy_sample = zeros(nG)

    # Kf = kernel_var  * Matern32Kernel()  #  ∘ ScaleTransform(kernel_scale ) 

    Kf = sekernel2(kernel_var, kernel_scale  )

        fkernal= Kf
        Yobs=YG
        
        Xobs=G[:,i]
        Xinducing=Gp[:,i]
        lambda=l2reg 
    
        Ko = kernelmatrix( fkernal, vec(Xobs) ) 
        Ki = kernelmatrix( fkernal, vec(Xinducing) ) 
        Kio = kernelmatrix( fkernal, vec(Xinducing), vec(Xobs) ) # transfer to inducing points
        Lo = cholesky(Symmetric( Ko + lambda*I)).L 
        Li = cholesky(Symmetric( Ki + lambda*I)).L   # cholesky on inducing locations  
        
        Gymu_sample  = Kio * ( Lo' \ (Lo \ Yobs ) )  # == mean_process mean latent process
        Gy_sample = Kio' * ( Li' \ (Li \ Gymu_sample ))  # mean process from inducing pts
        
        Gymu_sample  += Li * rand(Normal(0, 1), size(Li,1))   # faster sampling without covariance
        Gy_sample += Lo * rand(Normal(0, 1), size(Lo,2)) # error process 
    
        # sum_Gy_sample  = sum( Gymu_sample ) 
        # sum_Gy_sample  ~ Normal(0.0, 0.0001 * nInducing )  
        Ymu += Gy_sample
        # Gymu ~ fposterior(Xinducing, lambda)()  # a mechanism to store sampled mean process 
        
        # Main.DEBUG = fposterior(Xinducing, lambda)

        Y ~ arraydist( @. LogPoisson( Ymu ) )   
        
    return nothing
 
end
 