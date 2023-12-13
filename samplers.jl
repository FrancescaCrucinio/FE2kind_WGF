module samplers

using Statistics;
using StatsBase;
using IterTools;

export walker_sampler

# Walker's sampler for discrete probabilities
# OUTPUTS
# 1 - sampled indices of w
# INPUTS
# 'w' vector of weights (not normalised)
# 'M' number of samples
function walker_sampler(w, M)
    # lenght of w
    n = length(w);
    # probability and alias walker_matrix
    pm, am = walker_matrix(w);
    out = zeros(M, 1);
    unif = rand(M, 1);
    j = rand(1:n, M);
    out = ifelse.(unif .< pm[j], j, am[j]);
    out = trunc.(Int, out);
    return out
end
# Probability and aliasing matrix for Walker's sampler
# OUTPUTS
# 1 - probability matrix
# 2 - aliasing matrix
# INPUTS
# 'w' vector of weights (not normalised)
function walker_matrix(w)
    # lenght of w
    n = length(w);
    # normalise and multiply by length to get probability table
    w = n * w./sum(w);
    # overfull and underfull group
    overfull = findall(w .> 1);
    underfull = findall(w .< 1);
    alias_m = -ones(n);
    # make entries exactly full
    while (!isempty(overfull) & !isempty(underfull))
        j = pop!(underfull);
        i = overfull[end];
        alias_m[j] = i;
        w[i] = w[i] - 1 + w[j];
        if (w[i] < 1)
                append!(underfull, i);
                pop!(overfull);
        end
    end
    return w, alias_m
end
end