# Julia packages
push!(LOAD_PATH, "/Users/francescacrucinio/Documents/FE2kind_WGF")
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using RCall;
@rimport ks as rks;
using wgf2kind;
using RJmcmc_2kind;

# set seed
Random.seed!(1234);
# problem set up
lambda = 1/3;
lambda*(exp(1)-1)<1
K(x, y) = exp.(x .- y);
phi(x) = (1-lambda)*exp.(x)/(exp(1)-1);

### reversible jump MCMC
N = 200000;
c1_zero = (1-lambda);
@elapsed begin
X, k, p1 = RJMCMC_asset_pricing(N, phi, lambda, K);
end

x_values = range(0, 1, length = 100);
y_values = exp.(x_values)/(exp(1)-1);

pi_solution_mcmc = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_mcmc[i] = mean(K(x_values[i], getindex.(X,1)));
end
pi_solution_mcmc = lambda*pi_solution_mcmc*c1_zero/p1 .+ phi.(x_values);
plot(x_values, y_values)
plot!(x_values, pi_solution_mcmc)

