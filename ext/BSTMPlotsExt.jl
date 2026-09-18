module BSTMPlotsExt

using bstm
using Plots
using StatsPlots
using ColorSchemes
using DataFrames
using Statistics
using StatsBase
using LinearAlgebra
using Dates
using OrderedCollections
using Random
using DynamicPPL
using Turing
using Distributions
using NamedArrays
using Graphs

# Required functions for the extension to overload
import bstm: bstm_plots, _bstm_plots_impl, plot_spatial_surface, save_plots, plot_kde_simple, create_theme, choropleth, timeseries_ci, spatial_graph_plot, render_paths!, map_point_occupancy, save_plot, model_results_plots, plot_hsi_choropleth, plot_diffusion_map, plot_residence_time_map, plot_advection_arrows, plot_velocity_field, plot_hexagonal_field, plot_hydrodynamic_stratification, plot_hydrodynamic_diffusion, plot_hydrodynamic_section, par_credible_interval_plot, par_forest_plot, plot_ppc, plot_prior_vs_posterior, plot_marginal_effects, plot_spatial_residuals, plot_spatiotemporal_facets

# Internal functions from bstm that plotting.jl needs
import bstm: _detect_xy_columns, _detect_time_column, _detect_response_column, _detect_spatial_unit_column
import bstm: _generate_conditional_predictions
import bstm: AbstractModelArchitecture, UnivariateArchitecture, MultivariateArchitecture, MultifidelityArchitecture
import bstm: Mixed, RFF, SpectralGP, WaveletGP, SPDE, TPS, GP, Nystrom, SVC

# Now include the monolithic plotting file
include("../src/plotting.jl")

end # module BSTMPlotsExt
