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
lambda = 1/2;
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
alphas = [0.001 0.01 0.1 1 5];
x = zeros(Niter, Nparticles, length(alphas));
x_noref = zeros(Niter, Nparticles, length(alphas));
for i=1:length(alphas)
    x[:, :, i], x_noref[:, :, i] = wgf2kind_toy_gaussian_noreference(Nparticles, dt, Niter, alphas[i], x0, m0, sigma0, lambda);
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
function functional_wgf2kind_noref(piSample, lambda, alpha_param, phi, K)
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    loglik = zeros(length(piSample));
    for i=1:length(piSample)
        loglik[i] = mean(K.(piSample[i], piSample));
    end
    kl = mean(log.(pihat./(lambda*loglik.+phi.(piSample))));
    ent = mean(log.(pihat));
    return kl+alpha_param*ent;
end

EWGF = zeros(Niter, length(alphas));
EWGF_noref = zeros(Niter, length(alphas));
for j=1:length(alphas)
    for i=1:Niter
        EWGF[i, j] = functional_wgf2kind(x[i, :, j], lambda, alphas[j], m0, sigma0, phi, K);
        EWGF_noref[i, j] = functional_wgf2kind_noref(x_noref[i, :, j], lambda, alphas[j], phi, K);
    end
end
plot(1:Niter, EWGF)
plot!(1:Niter, EWGF_noref)

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

dx = x_values[2] - x_values[1];
dx*sum((y_values .- pi_solution_wgf[:, 1]).^2)
dx*sum((y_values .- pi_solution_wgf[:, 2]).^2)
dx*sum((y_values .- pi_solution_wgf[:, 3]).^2)
dx*sum((y_values .- pi_solution_wgf[:, 4]).^2)

dx*sum((y_values .- pi_solution_wgf_noref[:, 1]).^2)
dx*sum((y_values .- pi_solution_wgf_noref[:, 2]).^2)
dx*sum((y_values .- pi_solution_wgf_noref[:, 3]).^2)
dx*sum((y_values .- pi_solution_wgf_noref[:, 4]).^2)