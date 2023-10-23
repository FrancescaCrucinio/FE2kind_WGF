module RJmcmc_2kind;

using Distributions;
using Statistics;


export mcmc_weights

#= Weight computation for reversible jump MCMC
OUTPUTS
1 - weight values for the input trajectory
INPUTS
'k' length of trajectory
'x' trajextory path
'phi' forcing term
'lambda' eigenvalue
'K' integral kernel
=#
function mcmc_weights(k, x, phi, lambda, K)
    weight = phi(x[k]);
    for i=2:k
        weight = weight * (lambda * K(x[i-1], x[i]))
    end
    return weight
end

end