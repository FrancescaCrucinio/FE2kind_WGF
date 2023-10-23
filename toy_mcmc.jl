### reversible jump MCMC


N = 1000;
k = zeros(N);
    k = map(Int, k);
k[1] = 1;
X = [rand(1)];

p_birth = p_death = 1/3;

for n=2:N
    print("$n\n")
    if (k[n-1] > 1)
        pd = p_death;
    else
        pd = 0;
    end
    u = rand(1)[1];
    if (u < p_birth) # birth move
        print("birth attempted\n")
        j = Int(sample(1:(k[n-1]+1)));
        proposal_k = k[n-1]+1;
        proposal_X = cat(X[n-1][1:(j-1)], rand(1), X[n-1][j:k[n-1]], dims = 1);
        print("birth done\n")
    elseif (u < p_birth + pd) # death move
        print("death attemted\n")
        j = Int(sample(1:k[n-1]));
        proposal_k = k[n-1]-1;
        proposal_X = cat(X[n-1][1:(j-1)], X[n-1][(j+1):k[n-1]], dims = 1);
        print("death done\n")
    else # update move
        print("update attempted\n")
        k[n] = k[n-1];
        j = Int(sample(1:k[n]));
        proposal_k = k[n];
        proposal_X = cat(X[n-1][1:(j-1)], rand(1), X[n-1][(j+1):k[n-1]], dims = 1);
        print("update done\n")
    end
    # accept/reject
    p_accept = (mcmc_weights(Int(proposal_k), proposal_X, phi, lambda, K)/
            mcmc_weights(Int(k[n-1]), X[n-1], phi, lambda, K))
    u = rand(1)[1];
    if (u < p_accept)
        k[n] = proposal_k;
        push!(X, proposal_X)
    end
    k = map(Int, k);
end