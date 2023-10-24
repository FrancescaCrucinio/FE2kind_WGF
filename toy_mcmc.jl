### reversible jump MCMC


N = 100000;
k = zeros(N);
    k = map(Int, k);
k[1] = 1;
X = [rand(1)];

p_birth = p_death = 1/3;
count_zero_k = 1
for n=2:N
    print("$n\n")
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
c1 = -1 + 1/2*sqrt(pi)*erfi(1)
p1 = count_zero_k/N


# compute weights
W = zeros(N);
for n=1:N
    W[n] = asset_pricing_weights(k[n], X[n], phi, lambda, K, p_death)
end
x_values = range(0, 1, length = 100);
pi_solution_mcmc = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_mcmc[i] = phi(x_values[i]) + lambda*mean(W .* K(x_values[i], getindex.(X,1)))
end
plot(x_values, pi_solution_mcmc)

n = 3

phi(X[n][k[n]])/p_death
(lambda/(1-p_death))^k[n]