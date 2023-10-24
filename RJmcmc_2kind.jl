module RJmcmc_2kind;

using Distributions;
using Statistics;
using SpecialFunctions;

export asset_pricing_weights
export RJMCMC_asset_pricing

#= Reversible jump MCMC for asset pricing model
OUTPUTS
1 - trajectories
INPUTS
'N' number of trajectories
'phi' forcing term
'lambda' eigenvalue
'K' integral kernel
=#
function RJMCMC_asset_pricing(N, phi, lambda, K)
    k = zeros(N);
    k = map(Int, k);
    k[1] = 1;
    X = [rand(1)];

    p_birth = p_death = 1/3;
    count_zero_k = 1
    for n=2:N
        if (k[n-1] > 1)
            pd = p_death;
        else
            pd = 0;
        end
        u = rand(1)[1];
        if (u < p_birth) # birth move
            j = Int(sample(1:(k[n-1]+1)));
            proposal_k = k[n-1]+1;
            proposal_X = cat(X[n-1][1:(j-1)], rand(1), X[n-1][j:k[n-1]], dims = 1);
        elseif (u < p_birth + pd) # death move
            j = Int(sample(1:k[n-1]));
            proposal_k = k[n-1]-1;
            proposal_X = cat(X[n-1][1:(j-1)], X[n-1][(j+1):k[n-1]], dims = 1);
        else # update move
            k[n] = k[n-1];
            j = Int(sample(1:k[n]));
            proposal_k = k[n];
            proposal_X = cat(X[n-1][1:(j-1)], rand(1), X[n-1][(j+1):k[n-1]], dims = 1);
        end
        # accept/reject
        p_accept = (asset_pricing_acceptance_prob(Int(proposal_k), proposal_X, phi, lambda, K)/
            asset_pricing_acceptance_prob(Int(k[n-1]), X[n-1], phi, lambda, K))
        u = rand(1)[1];
        if (u < p_accept)
            k[n] = proposal_k;
            push!(X, proposal_X)
        else
            k[n] = k[n-1];
            push!(X, X[n-1])
        end
        k = map(Int, k);
        if (k[n] == 1)
            count_zero_k = count_zero_k + 1
        end
    end
    c1_zero = -1 + 1/2*sqrt(pi)*erfi(1)
    p1 = count_zero_k/N
    return X, k, c1_zero, p1
end

#= Acceptance probability for reversible jump MCMC
OUTPUTS
1 - weight values for the input trajectory
INPUTS
'k' length of trajectory
'x' trajectory path
'phi' forcing term
'lambda' eigenvalue
'K' integral kernel
=#
function asset_pricing_acceptance_prob(k, x, phi, lambda, K)
    weight = phi(x[k]);
    for i=2:k
        weight = weight * (lambda * K(x[i-1], x[i]))
    end
    return weight
end


#= Weights for asset pricing model 
OUTPUTS
1 - weight values for the input trajectory
INPUTS
'k' length of trajectory
'x' trajectory path
'phi' forcing term
'lambda' eigenvalue
'K' integral kernel
'p_death' probability of death
=#
function asset_pricing_weights(k, x, phi, lambda, K, p_death)
    weight = phi(x[k])/p_death;
    if (k > 1)
        weight = weight * (lambda/(1-p_death))^k
    end
    return weight
end

end