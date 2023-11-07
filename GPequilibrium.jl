# Julia packages
push!(LOAD_PATH, "/Users/francescacrucinio/Documents/FE2kind_WGF")
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using Revise;
using RCall;
@rimport ks as rks;
using wgf2kind;
using RJmcmc_2kind;

# set seed
Random.seed!(12345);
# data from SSM
m = 20;
f(x) = 0.01*x^3 - 0.2*x^2 + 0.2*x;
xs = 10*rand(m) .- 5;
zs = f.(xs) .+ randn(m);

cov_op(x, y) = exp(-(x - y)^2/(2*3.59^2))*4.21^2;
matrixK = zeros(m, m);
for i=1:m
    matrixK[i, :] = cov_op.(xs[i], xs);
end
matrixH = inv(matrixK + I(m));
# define kernel and forcing
function K(x, y)
    transition_mean = transpose(cov_op.(y, xs))*matrixH*zs;
    transition_variance = cov_op(y, y) - transpose(cov_op.(y, xs))*matrixH*cov_op.(y, xs);
    return(pdf.(Normal(transition_mean, transition_variance), x))
end

# check GP fit
x = range(-5, 5, length = 100);
gp_mean = zeros(100);
for i=1:100
    gp_mean[i] = transpose(cov_op.(x[i], xs))*matrixH*zs;
end
plot(x, f.(x))
scatter!(xs, zs)
plot!(x, gp_mean)


### WGF 
# parameters
alpha_param = 0.001;
m0 = 0;
sigma0 = 1;
# dt and number of iterations
dt = 1e-03;
Niter = 1000;
Nparticles = 100;
x = zeros(Niter, Nparticles);
# initial distribution is given as input:
x0 = rand(10*Normal.(0, 1) - 5, Nparticles);
@elapsed begin
x = wgf2kind_gpssm(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0, xs, zs);
end
# functional approximation
function functional_wgf2kind(piSample, lambda, alpha_param, m0, sigma0, K)
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    loglik = zeros(length(piSample));
    for i=1:length(piSample)
        loglik[i] = mean(K.(piSample[i], piSample));
    end
    kl = mean(log.(pihat./(lambda*loglik)));
    prior = pdf.(Normal(m0, sigma0), piSample);
    kl_prior = mean(log.(pihat./prior));
    return kl+alpha_param*kl_prior;
end

EWGF = zeros(Niter);
for i=1:Niter
    EWGF[i] = functional_wgf2kind(x[i, :], lambda, alpha_param, m0, sigma0, K);
end
plot(1:Niter, EWGF)

x_values = range(-12, 12, length = 100);
pi_solution_wgf = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_wgf[i] = mean(K(x_values[i], x[Niter, :]))
end
plot(x_values, pi_solution_wgf)

