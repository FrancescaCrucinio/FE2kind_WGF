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
beta = 0.5;
varK = 1-exp(-2*beta);
lambda = 10/11;
K(x, y) = pdf.(Normal(x*exp(-beta), sqrt(varK)), y);
phi(x) = (1-lambda)*pdf.(Normal(0, 1), x);
# parameters
m0 = 0;
sigma0 = 1;
# dt and number of iterations
dt = 1e-02;
Niter = 200;
Nparticles = 100;
# initial distribution is given as input:
x0 = rand(Normal.(0, 0.1), Nparticles);
# regularisation
alphas = [0.001 0.01 0.1];
x = zeros(Niter, Nparticles, length(alphas));
x_noref = zeros(Niter, Nparticles, length(alphas));
for i=1:length(alphas)
    x[:, :, i], x_noref[:, :, i] = wgf2kind_toy_gaussian_noreference(Nparticles, dt, Niter, alphas[i], x0, lambda);
end
# functional approximation
function functional_wgf2kind(piSample, lambda, alpha_param, m0, sigma0, phi, K)
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    loglik = zeros(length(piSample));
    for i=1:length(piSample)
        loglik[i] = mean(K.(piSample[i], piSample));
    end
    kl = mean(log.(pihat./(lambda*loglik.+phi.(piSample))));
    prior = pdf.(Normal(m0, sigma0), piSample);
    kl_prior = mean(log.(pihat./prior));
    return kl+alpha_param*kl_prior;
end

EWGF_alpha0 = zeros(Niter);
EWGF_alpha001 = zeros(Niter);
EWGF_alpha01 = zeros(Niter);
for i=1:Niter
    EWGF_alpha0[i] = functional_wgf2kind(x_alpha0[i, :], lambda, 0.0, m0, sigma0, phi, K);
    EWGF_alpha001[i] = functional_wgf2kind(x_alpha001[i, :], lambda, 0.001, m0, sigma0, phi, K);
    EWGF_alpha01[i] = functional_wgf2kind(x_alpha01[i, :], lambda, 10, m0, sigma0, phi, K);
end
plot(1:Niter, EWGF_alpha0)
plot!(1:Niter, EWGF_alpha001)
plot!(1:Niter, EWGF_alpha01)


### reversible jump MCMC
N = 60000;
@elapsed begin
c1_zero = (1-lambda);
X, k, p1 = RJMCMC_toy_gaussian(N, phi, lambda, K);
end

x_values = range(-5, 5, length = 100);
y_values = pdf.(Normal(0, 1), x_values);
pi_solution_wgf = zeros(length(x_values), length(alphas));
pi_solution_wgf_noref = zeros(length(x_values), length(alphas));

for j=1:length(alphas)
    for i=1:length(x_values)
        pi_solution_wgf[i, j] = phi(x_values[i]) + lambda*mean(K(x_values[i], x[Niter, :, j]))
        pi_solution_wgf_noref[i, j] = phi(x_values[i]) + lambda*mean(K(x_values[i], x_noref[Niter, :, j]))
    end
end
plot(x_values, y_values)
plot!(x_values, pi_solution_wgf)
plot!(x_values, pi_solution_wgf_noref)

pi_solution_mcmc = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_mcmc[i] = mean(K(x_values[i], getindex.(X,1)));
end
pi_solution_mcmc = lambda*pi_solution_mcmc*c1_zero/p1 .+ phi.(x_values);
plot!(x_values, pi_solution_mcmc)