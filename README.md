# Bayesian Spatial-temporal Models in Julia/Turing
 
 
## Installation

Clone the repository or copy/download the Julia files (\*.jl)

 
## Introduction

Bayesian spatial-temporal models are conceptually simple but operationally they are difficult as approximations are necessarily in almost any realistic situation. 

The situation usually encountered is data that are sparse and irregularly captured in time and space, methods that require high computational requirements or specialized knowledge, and a plethora of fragmented (black-box) applications and approaches.  

The useful answer to the above is to adopt a Bayesian framework. Though it increases computational and conceptual complexity, the assumptions and implementation of methods become almost completely transparent. Here we address many families of spatial-temporal models and show how they are related and useful in various contexts.  

Two main classes can be seen. A discrete form and a continuous form. Each are examined using the Julia and Turing framework. Both are high performance language frameworks that can help with many aspects of our problem. Here you will find a number of Julia functions that can help address these problems and being completely transparent, open to modification and enhancement by subject matter specialists and hobbyists.  


## Conditional AutoRegressive (CAR) Space-Time Models (in Julia) and other approaches:

CAR models are discrete and just about the simplest possible way of accounting for spatial autorcorrelation. It is essentially the analogue of the temporal ARIMA type models but in space. It is arguably even simpler as the units of space are discrete areal units (arbitrary polygons). Originally it was formulated in lattice form from the Ising models in physics. Its use is most prevalent in epidemiology made accessible by the original Besag-York-Mollie paper, Leroux and Riebler, and many others.

For the R implementation (front-end to INLA): see https://github.com/jae0/carstm .

* [https://github.com/jae0/carstm/inst/scripts/example\_temperature\_carstm.md](https://github.com/jae0/carstm/inst/scripts/example_temperature_carstm.md)


* CAR/ICAR/BYM areal unit models (space)
* Non-intrinsic, localised, and MCAR variants developed in CARBayes R-library 

Most didactic are the examples that use STAN (see info copied from various sources in "docs/example\_car\_stan.md").

The INLA/R, BRMS/R, Turing/Julia (subdirectory examples) implementations are confirming parameter estimates.

Ultimately, we use Julia/Turing due to simplicity and speed and attempt to develop a useful statistical and dynamical model of snow crab spatiotemporal processes.

It is similar in scope to the github.com/jae0/carstm, however, with no reliance upon INLA for computation.

See examples and progression of model building in [./docs/](./docs/):

## Continuous models
 
* Kriging is a well known technique. It is a least-squares (~ Gaussian) representation of spatial autocorrelation structure, used in many fields. 
* Spatio-temporal GP models are the more formal way of representing these processes covariates ("features"). They are wonderfully simple but horribly expensive to compute. 
* Representation/approximation of these processes (Sparse GPs, Spectral methods) require specialized knowledge that is often not easily accessible. We try to make amends here.   


## More complex models

* Spacetime models that combine all of the above together with an aggregate dynamical (biological/ecological) process and/or a spatial or spatiotemporal process (e.g., movement, differential survival, etc.) all become tantalizingly close to being accessible, especially with the large dynamical modelling possibilities in Julia/SciML. This moves us a bit closer to this goal.
 


## Useful references

Mitzi Morris: https://mc-stan.org/users/documentation/case-studies/icar\_stan.html (very thorough)

Max Joseph: Exact sparse CAR models in Stan: https://github.com/mbjoseph/CARstan

https://github.com/ConnorDonegan/Stan-IAR
 
https://www.mdpi.com/1660-4601/18/13/6856

Besag, Julian, Jeremy York, and Annie Mollié. "Bayesian image restoration, with two applications in spatial statistics." Annals of the institute of statistical mathematics 43.1 (1991): 1-20.

Gelfand, Alan E., and Penelope Vounatsou. "Proper multivariate conditional autoregressive models for spatial data analysis." Biostatistics 4.1 (2003): 11-15.

Jin, Xiaoping, Bradley P. Carlin, and Sudipto Banerjee. "Generalized hierarchical multivariate CAR models for areal data." Biometrics 61.4 (2005): 950-961.

