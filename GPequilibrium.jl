# Julia packages
push!(LOAD_PATH, "/Users/francescacrucinio/Documents/FE2kind_WGF")
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using Revise;
using LaTeXStrings
using RCall;
@rimport ks as rks;
using wgf2kind;

# set seed
Random.seed!(12345);
# data from SSM
f(x) = 0.01*x^3 - 0.2*x^2 + 0.2*x;
noise = 5;
m = 20;
input = 10*rand(m) .- 5;
output = f.(input).+ noise*randn(m);
cov_op(x, y) = exp(-(x - y)^2/(2*3.59^2))*4.21^2;

matrixK = zeros(m, m);
for i=1:m 
    for j=1:m
        matrixK[i, j] = cov_op(input[i], input[j]);
    end
end
invK = inv(matrixK + noise*I(m));
# check GP fit
x = range(-12, 8, length = 100);
gp_mean = zeros(100);
gp_variance = zeros(100);
for i=1:100
    gp_mean[i] = transpose(cov_op.(x[i], input))*invK*output;
    gp_variance[i] = cov_op(x[i], x[i]) - transpose(cov_op.(x[i], input))*invK*cov_op.(x[i], input);
end
# plot(x, f.(x))
scatter(input, output)
plot!(x, gp_mean)
plot!(x, gp_mean+gp_variance, linestyle=:dash, color = :gray)
plot!(x, gp_mean-gp_variance, linestyle=:dash, color = :gray)

# define kernel and forcing
function K_mean(y) 
    return(transpose(cov_op.(y, input))*invK*output)
end
function K_variance(x) 
    k_vector = cov_op.(x, input);
    return(cov_op(x, x) - transpose(k_vector)*invK*k_vector)
end 
function K(x, y)
    transition_mean = K_mean(y);
    transition_variance = K_variance(y);
    return(pdf.(Normal(transition_mean, transition_variance), x))
end

### Nystrom
q = 499;
xq = range(-20, 10, length = q+1);
dx = xq[2] - xq[1];
M = zeros(q+1, q+1);
for i=1:(q+1)
    for j=1:(q+1)
        M[i, j] = -dx*K(xq[i], xq[j]);
    end
end
M[:, 1] = M[:, 1]/2;
M[:, q+1] = M[:, q+1]/2;
M = M + I(q+1);

Mp = [M; dx*ones(1, q+1)];
Mp[q+2, 1] = Mp[q+2, 1]/2;
Mp[q+2, q+1] = Mp[q+2, q+1]/2;
bp = zeros(1, q+2);
bp[q+2] = 1;

u = Mp \ transpose(bp);

### WGF 
# parameters
alpha_param = 0.001;
m0 = 0;
sigma0 = 1;
# dt and number of iterations
Niter = 100;
Nparticles = 200;
dt = 5e-03;
x = zeros(Niter, Nparticles);
# initial distribution is given as input:
x0 = rand(Normal.(0, 5), Nparticles);
@elapsed begin
x = wgf2kind_gpssm(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0, input, output);
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
    pi_solution_wgf[i] = mean(K.(xq[i], x[Niter, :]))
end

p1 = plot(xq, u, label = "Nystrom", lw = 3, linestyle = :dash, color = :gray)
plot!(p1, xq, pi_solution_wgf, label = "FE2kind-WGF", 
lw = 3, linestyle = :dot, color = :gray, legendfontsize = 10, legend=:topleft)
# savefig(p1, "gp_ssm.pdf")

# check invariance
M = 20000;
x_new_wgf = zeros(M);
x_new_nystrom = zeros(M);
for i=1:M
    index_wgf = sample(1:Nparticles)
    x_new_wgf[i] = K_mean(x[Niter, index_wgf]) + sqrt(K_variance(x[Niter, index_wgf]))*randn(1)[1];
    index_nystrom = walker_sampler(u[:], 1);
    x_new_nystrom[i] = K_mean(xq[index_nystrom][1]) + sqrt(K_variance(xq[index_nystrom][1]))*randn(1)[1];
end


p2 = histogram(x_new, normalize = :pdf, bins = 200, color = :gray, label = "FE2kind-WGF", alpha = 0.3)
histogram!(p2, x_new_nystrom, normalize = :pdf, bins = 200, color = :red, label = "Nystrom", alpha = 0.2)
plot!(p2, xq, pi_solution_wgf, lw = 3, linestyle = :dashdot, color = :black, label = "FE2kind-WGF",
legendfontsize = 10, legend=:topleft)
plot!(p2, xq, u, lw = 3, linestyle = :dashdot, color = :red, label = "Nystrom",
legendfontsize = 10, legend=:topleft)
# savefig(p2, "gp_ssm_histogram.pdf")
