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
using Roots;
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

cost = [50 100 200 500 1000];
mse_nystrom = zeros(length(cost));
Nrep = 100;
mse_wgf = zeros(length(cost), Nrep);

for c=1:length(cost)
    ### Nystrom
    q = cost[c];
    xq = range(-a, a, length = q);
    dx = xq[2] - xq[1];
    Kmatrix = zeros(q, q);
    for i=1:q
        for j=1:q
            Kmatrix[i, j] = K(xq[i], xq[j]);
        end
    end
    Kmatrix[:, 1] = Kmatrix[:, 1]/2;
    Kmatrix[:, q] = Kmatrix[:, q]/2;

    eigvalues = eigvals(Kmatrix);
    eigvectors = eigvecs(Kmatrix);
    eig_index = q;
    lambda_nystrom = real(eigvalues[eig_index])*dx;
    solution_nystrom = abs.(real.(eigvectors[:, eig_index]))/sqrt(dx);
    mse_nystrom[c] = dx*sum((solution_nystrom -  eigenfun.(xq)).^2);
    for j=1:Nrep
        ### WGF 
        # parameters
        alpha_param = 0.01;
        m0 = 0;
        sigma0 = 0.05;
        # dt and number of iterations
        Niter = 400;
        Nparticles = cost[c];
        dt = 1/cost[c];
        x = zeros(Niter, Nparticles);
        # initial distribution is given as input:
        x0 = rand(Normal.(0, 0.05), Nparticles);
        x = wgf2kind_kl_exp(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0, lambda);
        pi_solution_wgf = zeros(length(xq));
        for i=1:length(xq)
            pi_solution_wgf[i] = lambda*mean(K.(xq[i], x[Niter, :]));
        end
        mse_wgf[c, j] = dx*sum((pi_solution_wgf -  eigenfun.(xq)).^2);
    end
end




gain = mse_nystrom./mse_wgf;
bp = boxplot(cost, transpose(gain), xaxis = :log10, outliers = false,
    legend = :none, bar_width = 10, range = 0, tickfontsize = 15, color = :gray)
# savefig(bp, "kl_boxplot.pdf")