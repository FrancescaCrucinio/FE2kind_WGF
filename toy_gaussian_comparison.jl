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
lambda = 2/3;
K(x, y) = pdf.(Normal(x*exp(-beta), sqrt(varK)), y);
phi(x) = (1-lambda)*pdf.(Normal(0, 1), x);
# parameters
alpha_param = 0.001;
m0 = 0;
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-03;
Niter = 1000;
Nparticles = 100;
Npaths = 50000;
c1_zero = (1-lambda);

# true solution
x_values = range(-5, 5, length = 100);
y_values = pdf.(Normal(0, 1), x_values);
dx = x_values[2] - x_values[1];

Nrep = 10;
# diagnostics
tRJ = zeros(Nrep, 1);
iseRJ = zeros(Nrep, 1);
meanRJ = zeros(Nrep, 1);
varRJ = zeros(Nrep, 1);
tWGF = zeros(Nrep, 1);
iseWGF = zeros(Nrep, 1);
meanWGF = zeros(Nrep, 1);
varWGF = zeros(Nrep, 1);
for j=1:Nrep
    # WGF
    x0 = rand(Normal.(0, 0.1), Nparticles);
    tWGF[j] = @elapsed begin
    x = wgf2kind_toy_gaussian(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0);
    end
    meanWGF[j] = mean(x[Niter, :]);
    varWGF[j] = var(x[Niter, :]);
    pi_solution_wgf = zeros(length(x_values));
    for i=1:length(x_values)
        pi_solution_wgf[i] = phi(x_values[i]) + lambda*mean(K(x_values[i], x[Niter, :]))
    end
    iseWGF[j] = dx*sum((y_values .- pi_solution_wgf).^2);
    # RJMCMC
    tRJ[j] = @elapsed begin
    X, k, p1 = RJMCMC_toy_gaussian(Npaths, phi, lambda, K);
    end
    meanRJ[j] = c1_zero/p1*mean(getindex.(X,1));
    varRJ[j] = c1_zero/p1*mean(getindex.(X,1).^2) - meanRJ[j]^2;
    pi_solution_mcmc = zeros(length(x_values));
    for i=1:length(x_values)
        pi_solution_mcmc[i] = mean(K(x_values[i], getindex.(X,1)));
    end
    pi_solution_mcmc = lambda*pi_solution_mcmc*c1_zero/p1 .+ phi.(x_values);
    iseRJ[j] = dx*sum((y_values .- pi_solution_mcmc).^2);
end