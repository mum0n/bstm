---
title: "Creating a Custom Component in bstm"
format: html
---

# Creating a Custom Component in `bstm`

## 1. Introduction

The `bstm` framework is designed to be extensible. The core of this extensibility lies in the `ComponentModel` interface. By implementing a set of five standard functions for a new component type, you can seamlessly integrate custom statistical models into the `bstm` formula parser, code generator, and post-processing engine.

This tutorial will walk you through the process of creating a simple custom component: a non-linear power-law function of a covariate. The model we want to implement is:

$$ f(x) = a \cdot x^b $$

where `a` (amplitude) and `b` (exponent) are parameters we want to estimate. We will create a new component called `PowerLaw` that can be used in a `bstm` formula like this:

```julia
m = @bstm(likelihood(y) ~ random(x, model=:powerlaw), data)
```

## 2. The `ComponentModel` Interface

Any new component must implement the following five methods:

1.  **`get_datastructures!`**: Validates data and sets up model-wide configurations.
2.  **`get_precomputes`**: Performs data-independent calculations before sampling.
3.  **`get_priors`**: Generates the Turing code for the component's priors.
4.  **`get_updates`**: Generates the Turing code to construct the latent effect and add it to the linear predictor.
5.  **`get_effects`**: Reconstructs the posterior effect from the MCMC chain.

Let's implement these for our `PowerLaw` component.

## 3. Step-by-Step Implementation

### Step 3.1: Define the Component Struct

First, we define a struct for our component. It must subtype `ComponentModel`. Its fields will store the prior distributions for the parameters `a` and `b`.

```julia
using Distributions

"""
    PowerLaw <: ComponentModel

A component for a non-linear power-law effect of a covariate, f(x) = a * x^b.
"""
struct PowerLaw <: ComponentModel
    a::UnivariateDistribution  # Prior for the amplitude 'a'
    b::UnivariateDistribution  # Prior for the exponent 'b'
end
```

### Step 3.2: Register the Component

To make `bstm` aware of our new component, we must register its type and its constructor function.

```julia
# Register the type so the parser can find it.
COMPONENT_TYPE_REGISTRY[:powerlaw] = PowerLaw

# Register the constructor. This function takes the resolved hyperpriors
# and any parameters from the formula call and returns an instance of our struct.
COMPONENT_CONSTRUCTORS[:powerlaw] = (p, params) -> PowerLaw(p.a, p.b)

# Tell the framework that this is a 'smooth' type component.
MODEL_TO_STRUCTURE_MAP[:powerlaw] = :smooth
```

The `resolve_hyperpriors` function (part of the `bstm` engine) will automatically look for priors named `:a` and `:b` in the user's `hyperpriors` dictionary or use defaults. For this example, we'll assume defaults are defined elsewhere, or the user provides them.

### Step 3.3: Implement `get_datastructures!`

This function checks if the covariate `x` exists in the data. For this simple component, that's all we need.

```julia
function get_datastructures!(m_type::Type{PowerLaw}, M::Dict, mod_data::Dict)::Bool
    # `mod_data[:variables]` will contain the variable passed to `random()`, e.g., `[:x]`.
    if isempty(mod_data[:variables])
        error("The PowerLaw model requires a covariate, e.g., `random(x, model=:powerlaw)`.")
    end

    covariate_name = mod_data[:variables]
    if !hasproperty(M[:data], covariate_name)
        error("Covariate ':$covariate_name' for PowerLaw model not found in data.")
    end

    # This component doesn't need to modify the main M dictionary,
    # so we just return true to indicate it should be included.
    return true
end
```

### Step 3.4: Implement `get_precomputes`

Our `PowerLaw` component does not require any pre-computation (like building a basis matrix). So, this function simply returns an empty `NamedTuple`.

```julia
function get_precomputes(m::PowerLaw, M::NamedTuple, mod_data::Dict)::NamedTuple
    return (;) # Return an empty NamedTuple
end
```

### Step 3.5: Implement `get_priors`

This function generates the Turing code for sampling the parameters `a` and `b`. We use the `generate_full_variable_names` utility to create unique names for our parameters to avoid collisions.

```julia
function get_priors(m::PowerLaw, spec::NamedTuple, arch::String, outcome_idx, M)::String
    # `spec.key` provides the unique key for this component instance.
    # `generate_full_variable_names` creates names like `a_powerlaw_x` and `b_powerlaw_x`.
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix="param")

    # `_distribution_to_string` is an internal bstm utility to convert a Distribution to code.
    prior_a_str = _distribution_to_string(m.a)
    prior_b_str = _distribution_to_string(m.b)

    return """
    $(v.a) ~ $(prior_a_str)
    $(v.b) ~ $(prior_b_str)
    """
end
```

### Step 3.6: Implement `get_updates`

This function generates the code to calculate the effect `a * x^b` and add it to the linear predictor `eta`.

```julia
function get_updates(m::PowerLaw, spec::NamedTuple, arch::String, outcome_idx, M)::String
    v_params = generate_full_variable_names(spec, arch, outcome_idx; prefix="param")
    v_latent = generate_full_variable_names(spec, arch, outcome_idx)
    
    covariate_name = spec.params[:positional_args]
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    return """
    # --- PowerLaw Component: $(spec.key) ---
    local covariate_data = M.data[!, :$(covariate_name)]
    $(v_latent.latent) = $(v_params.a) .* (covariate_data .^ $(v_params.b))
    $(eta_target) .+= $(v_latent.latent)
    """
end
```

### Step 3.7: Implement `get_effects`

This final function reconstructs the posterior effect. It extracts the posterior samples for `a` and `b` from the MCMC chain and recalculates the effect for each sample.

```julia
function get_effects(m::PowerLaw, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = []
    covariate_name = spec.params[:positional_args]
    covariate_data = M.data[!, covariate_name]

    for k in 1:outcomes_N
        v_params = generate_full_variable_names(spec, M.model_arch, k; prefix="param")
        
        # Extract posterior samples from the chain
        a_samples = get_params_vector(chain, string(v_params.a), 1)
        b_samples = get_params_vector(chain, string(v_params.b), 1)

        effect_k = Matrix{Float64}(undef, M.y_N, n_samples)

        for s in 1:n_samples
            a_s = a_samples[s, 1]
            b_s = b_samples[s, 1]
            effect_k[:, s] = a_s .* (covariate_data .^ b_s)
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
```

## 4. Conclusion

By implementing these five methods, we have created a new, fully functional `PowerLaw` component. It can now be used in any `bstm` model formula, and the framework will automatically handle its integration into the model, sampling, and post-processing. This modular design makes `bstm` a powerful tool for rapid prototyping and development of custom statistical models.