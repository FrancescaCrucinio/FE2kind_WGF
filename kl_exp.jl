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
K(x, y) = exp.(-abs(y .- x));
a = 1;

### analytic solution 
f(x) = 1 - x*tan(x*a);
omega = find_zeros(f, -a, a);
lambda = unique(2 ./(1 .+ omega.^2))[1]
eigenfun(x) = cos(lambda*x)/sqrt(a+sin(2*a*lambda)/(2*lambda));


### Nystrom
q = 100;
xq = range(-a, a, length = q+1);
dx = xq[2] - xq[1];
Kmatrix = zeros(q+1, q+1);
for i=1:(q+1)
    for j=1:(q+1)
        Kmatrix[i, j] = K(xq[i], xq[j]);
    end
end
Kmatrix[:, 1] = Kmatrix[:, 1]/2;
Kmatrix[:, q+1] = Kmatrix[:, q+1]/2;

eigvalues = eigvals(Kmatrix);
eigvectors = eigvecs(Kmatrix);

# set eigenvalue
eig_index = q+1;
lambda_nystrom = real(eigvalues[eig_index])*dx;
solution_nystrom = abs.(real.(eigvectors[:, eig_index]))/sqrt(dx);

### WGF 
# parameters
alpha_param = 0.001;
m0 = 0;
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-03;
Niter = 200;
Nparticles = 200;
x = zeros(Niter, Nparticles);
# initial distribution is given as input:
x0 = rand(Normal.(0, 0.1), Nparticles);
@elapsed begin
x = wgf2kind_kl_exp(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0, lambda);
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

pi_solution_wgf = zeros(length(xq));
for i=1:length(xq)
    pi_solution_wgf[i] = lambda*mean(K.(xq[i], x[Niter, :]))
end

plot(xq, eigenfun.(xq), label = "analytic", lw = 3, linestyle = :solid, color = :black)
plot!(xq, solution_nystrom, label = "Nystrom", lw = 3, linestyle = :dash, color = :gray)
plot!(xq, pi_solution_wgf, label = "FE2kind-WGF", lw = 3, linestyle = :dot, color = :gray, legendfontsize = 8, legend=:topright)


p1 = plot(xq, eigenfun.(xq)/sum(eigenfun.(xq)), label = "analytic", 
lw = 3, linestyle = :solid, color = :black)
plot!(p1, xq, solution_nystrom/sum(solution_nystrom), label = "Nystrom", 
lw = 3, linestyle = :dash, color = :gray)
plot!(p1, xq, pi_solution_wgf/sum(pi_solution_wgf), label = "FE2kind-WGF", 
lw = 3, linestyle = :dot, color = :gray, legendfontsize = 8, legend=:topright)
# savefig(p1, "kl_exponential.pdf")
