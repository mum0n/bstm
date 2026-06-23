module bstm

# # 1. Core Dependencies
using Turing, Distributions, LinearAlgebra, SparseArrays
using DataFrames, StatsModels, CategoricalArrays, StatsBase
using KernelFunctions, Distances, PDMats
using FlexiChains, MCMCChains, PosteriorStats
using DynamicPPL, AbstractPPL, Bijectors
using LogExpFunctions, SpecialFunctions
using FFTW
using Optim, ForwardDiff
using Requires

# # 2. Public API Exports

# Primary modeling and analysis functions
export bstm,
       predict,
       model_results_comprehensive,
       bstm_loo,
       compare_manifolds,
       get_optimal_sampler,
       get_inits

# Data utilities
export scottish_lip_cancer_data_spacetime,
       generate_sim_data

# Manifold struct exports for programmatic model building
# Discrete & Graph Primitives
export Fixed, IID, ICAR, Besag, BYM2, Leroux, SAR, RW1, RW2, AR1, DAG, NetworkFlow

# Continuous & Spectral Primitives
export GP, FITC, RFF, FFT, SPDE, SVGP, Warp, Nystrom

# Non-Euclidean & Decay Primitives
export ExponentialDecay, Hyperbolic

# Basis-Function & Spacetime Primitives
export Advection, Diffusion, AdvectionDiffusion, ST_I, ST_II, ST_III, ST_IV,
       TPS, BSpline, PSpline, Wavelets

# Seasonal & Periodic Primitives
export Harmonic, Cyclic

# Specialized & Network Manifolds
export Eigen, BCGN, LocalAdaptive

include("spatiotemporal_functions.jl")
include("bstm_modular.jl")

end # module bstm