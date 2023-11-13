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

# set seed
Random.seed!(1234);
# problem set up
K(x, y) = exp.(-(y .- x).^2/4);

# find eigenvalues and eigenvectors of corresponding matrix
q = 100;
xq = range(-1, 1, length = q+1);
dx = xq[2] - xq[1];
Kmatrix = zeros(q+1, q+1);
for i=1:(q+1)
    for j=1:(q+1)
        Kmatrix[i, j] = dx*K(xq[i], xq[j]);
    end
end
Kmatrix[:, 1] = Kmatrix[:, 1]/2;
Kmatrix[:, q+1] = Kmatrix[:, q+1]/2;

eigvalues = eigvals(Kmatrix);
eigvectors = eigvecs(Kmatrix);

# set eigenvalues
lambda = real(eigvalues[q+1]);

### WGF 
# parameters
alpha_param = 0.001;
m0 = 0;
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-03;
Niter = 100;
Nparticles = 100;
x = zeros(Niter, Nparticles);
# initial distribution is given as input:
x0 = rand(Normal.(0, 0.1), Nparticles);
@elapsed begin
x = wgf2kind_kl_squared_exp(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0, lambda);
end
# functional approximation
function functional_wgf2kind(piSample, alpha_param, m0, sigma0, K)
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    loglik = zeros(length(piSample));
    for i=1:length(piSample)
        loglik[i] = mean(K.(piSample[i], piSample));
    end
    kl = mean(log.(pihat./loglik));
    prior = pdf.(Normal(m0, sigma0), piSample);
    kl_prior = mean(log.(pihat./prior));
    return kl+alpha_param*kl_prior;
end

EWGF = zeros(Niter);
for i=1:Niter
    EWGF[i] = functional_wgf2kind(x[i, :], alpha_param, m0, sigma0, K);
end
plot(1:Niter, EWGF)

x_values = range(-1, 1, length = 100);
pi_solution_wgf = zeros(length(x_values));
for i=1:length(x_values)
    pi_solution_wgf[i] = mean(K.(x_values[i], x[Niter, :]))
end
plot(x_values, pi_solution_wgf/sum(pi_solution_wgf))



### Nystrom
M = -Kmatrix + I(q+1)/lambda;

Mp = [M; dx*ones(1, q+1)];
Mp[q+2, 1] = Mp[q+2, 1]/2;
Mp[q+2, q+1] = Mp[q+2, q+1]/2;
bp = zeros(1, q+2);
bp[q+2] = 1;

u = Mp \ transpose(bp);
plot!(xq, u/sum(u))
plot!(xq, real.(eigvectors[:, q+1])/sum(real.(eigvectors[:, q+1])))