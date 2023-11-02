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
K(x, y) = pdf.(Normal(x*exp(-beta), sqrt(varK)), y);
# parameters
alpha_param = 0.001;
m0 = 0;
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-03;
Niter = 1000;
Nparticles = 100;
Npaths = 60000;

# true solution
x_values = range(-5, 5, length = 100);
y_values = pdf.(Normal(0, 1), x_values);
dx = x_values[2] - x_values[1];

Nrep = 5;
lambdas = range(0.1, 1, length = 10);
Nlambdas = length(lambdas);
# diagnostics
tRJ = zeros(Nrep, Nlambdas);
iseRJ = zeros(Nrep, Nlambdas);
meanRJ = zeros(Nrep, Nlambdas);
varRJ = zeros(Nrep, Nlambdas);
tWGF = zeros(Nrep, Nlambdas);
iseWGF = zeros(Nrep, Nlambdas);
meanWGF = zeros(Nrep, Nlambdas);
varWGF = zeros(Nrep, Nlambdas);
for i=1:Nlambdas
    lambda = lambdas[i];
    phi(x) = (1-lambda)*pdf.(Normal(0, 1), x);
    c1_zero = (1-lambda);
    for j=1:Nrep
        # WGF
        x0 = rand(Normal.(0, 0.1), Nparticles);
        tWGF[j, i] = @elapsed begin
        x = wgf2kind_toy_gaussian(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0);
        end
        meanWGF[j, i] = mean(x[Niter, :]);
        varWGF[j, i] = var(x[Niter, :]);
        pi_solution_wgf = zeros(length(x_values));
        for n=1:length(x_values)
            pi_solution_wgf[n] = phi(x_values[n]) + lambda*mean(K(x_values[n], x[Niter, :]))
        end
        iseWGF[j, i] = dx*sum((y_values .- pi_solution_wgf).^2);
        # RJMCMC
        tRJ[j, i] = @elapsed begin
        X, k, p1 = RJMCMC_toy_gaussian(Npaths, phi, lambda, K);
        end
        meanRJ[j, i] = c1_zero/p1*mean(getindex.(X,1));
        varRJ[j, i] = c1_zero/p1*mean(getindex.(X,1).^2) - meanRJ[j]^2;
        pi_solution_mcmc = zeros(length(x_values));
        for n=1:length(x_values)
            pi_solution_mcmc[n] = mean(K(x_values[n], getindex.(X,1)));
        end
        pi_solution_mcmc = lambda*pi_solution_mcmc*c1_zero/p1 .+ phi.(x_values);
        iseRJ[j, i] = dx*sum((y_values .- pi_solution_mcmc).^2);
    end
end

plot(lambdas, mean(iseRJ, dims = 1)[:])
plot!(lambdas, mean(iseWGF, dims = 1)[:])