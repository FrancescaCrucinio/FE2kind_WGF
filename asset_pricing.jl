push!(LOAD_PATH, "/Users/francescacrucinio/Documents/FE2kind_WGF")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using SpecialFunctions;
using RCall;
@rimport ks as rks;
using wgf2kind;
using RJmcmc_2kind;

# set seed
Random.seed!(1234);

# problem set up 
lambda = 0.85;
a = 0.05;
b = 0.9;
varK = 100;
# define kernel and forcing
function K(x, y)
    d0 = Normal(a*x + b, sqrt(varK))
    truncated_d0 = truncated(d0, 0, 1)
    return(pdf.(truncated_d0, y))
end
phi(x) = (exp(x^2)-1) * (0 <= x <= 1);
phi_gradient(x) = 2*x*exp(x^2);

### reversible jump MCMC
N = 100000;
c1_zero = -1 + 1/2*sqrt(pi)*erfi(1);
@elapsed begin
    X, k, p1 = RJMCMC_asset_pricing(N, phi, lambda, K);
end

### WGF
alpha_param = 0.01;
m0 = 0.5;
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-02;
Niter = 20;
Nparticles = 10000;
x0 = rand(Normal.(0.5, 0.01), Nparticles);
@elapsed begin
    x = wgf2kind_asset_pricing(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0);
end

# check convergence
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

EWGF = zeros(Niter);
for i=1:Niter
    EWGF[i] = functional_wgf2kind(x[i, :], lambda, alpha_param, m0, sigma0, phi, K);
end
plot(1:Niter, EWGF)

# plot pi
RKDEyWGF = rks.kde(x = x[Niter, :]);
KDEx = rcopy(RKDEyWGF[2]);
KDEy = abs.(rcopy(RKDEyWGF[3]));
plot(KDEx, KDEy)

# get smooth representations
x_values = range(0, 1, length = 100);
dx = x_values[2] - x_values[1];

pi_solution_wgf = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_wgf[i] = mean(K(x_values[i], x[Niter, :]));
end
in01 = (0 .<= x[Niter, :] .<= 1);
pi_solution_wgf_normalized = lambda*pi_solution_wgf/(sum(in01)/Nparticles) + phi.(x_values);
pi_solution_wgf = lambda*pi_solution_wgf + phi.(x_values);

pi_solution_mcmc = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_mcmc[i] = mean(K(x_values[i], getindex.(X,1)));
end
pi_solution_mcmc_normalized = lambda*pi_solution_mcmc + phi.(x_values);
pi_solution_mcmc = lambda*pi_solution_mcmc*c1_zero/p1 + phi.(x_values);

# check match between empirical and smooth
plt1 = histogram(getindex.(X,1), bins = 30, normalize=:density, 
    weights = c1_zero/p1*ones(N)/N, color=:gray, alpha = 0.5)
plot!(plt1, x_values, pi_solution_mcmc, color=:black, lw = 3, legend = false)
# savefig(plt1, "asset_pricing_rjmcmc.pdf")

plt2 = histogram(x[Niter, in01], bins = 30, normalize=:pdf, 
    weights = Nparticles/sum(in01)*ones(sum(in01)), color=:gray, alpha = 0.5)
plot!(plt2, x_values, pi_solution_wgf, color=:black, lw = 3, legend = false)
# savefig(plt2, "asset_pricing_wgf.pdf")

# normalized plots
plt3 = plot(x_values, pi_solution_mcmc_normalized, lw = 3, 
    color=:blue, label = "RJ-MCMC", linestyle = :dash)
plot!(plt3, x_values, pi_solution_wgf_normalized, lw = 3, 
    color=:red, label = "FE2kind-WGF", linestyle = :dot, legendfontsize = 15, legend=:topleft)
# savefig(plt3, "asset_pricing_comparison.pdf")