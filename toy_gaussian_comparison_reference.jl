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
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-02;
Niter = 200;
Nparticles = 100;

# true solution
x_values = range(-5, 5, length = 100);
y_values = pdf.(Normal(0, 1), x_values);
dx = x_values[2] - x_values[1];

Nrep = 100;
# regularisation
alphas = range(0, 1, length = 10);
Nalphas = length(alphas);
# diagnostics
ise_noref = zeros(Nrep, Nalphas);
mean_noref = zeros(Nrep, Nalphas);
var_noref = zeros(Nrep, Nalphas);
ise_ref = zeros(Nrep, Nalphas);
mean_ref = zeros(Nrep, Nalphas);
var_ref = zeros(Nrep, Nalphas);
for i=1:Nalphas
    alpha_param = alphas[i];
    for j=1:Nrep
        print("$i, $j\n")
        # WGF
        x0 = rand(Normal.(0, 0.1), Nparticles);
        x, x_noref = wgf2kind_toy_gaussian_noreference(Nparticles, dt, Niter, alpha_param, x0, m0, sigma0, lambda);
        mean_ref[j, i] = mean(x[Niter, :]);
        var_ref[j, i] = var(x[Niter, :]);
        mean_noref[j, i] = mean(x_noref[Niter, :]);
        var_noref[j, i] = var(x_noref[Niter, :]);
        pi_solution_wgf = zeros(length(x_values));
        pi_solution_wgf_noref = zeros(length(x_values));
        for n=1:length(x_values)
            pi_solution_wgf[n] = phi(x_values[n]) + lambda*mean(K(x_values[n], x[Niter, :]))
            pi_solution_wgf_noref[n] = phi(x_values[n]) + lambda*mean(K(x_values[n], x_noref[Niter, :]))
        end
        ise_ref[j, i] = dx*sum((y_values .- pi_solution_wgf).^2);
        ise_noref[j, i] = dx*sum((y_values .- pi_solution_wgf_noref).^2);
    end
end

plt1 = plot(alphas[:], mean(ise_ref, dims = 1)[:], label = "Ref", 
    lw = 5, linestyle = :dash, color = :gray)
plot!(plt1, alphas[:], mean(ise_noref, dims = 1)[:], label = "No ref", 
    legendfontsize = 15, legend=:topleft, lw = 3, linestyle = :dot, color = :black)
# savefig(plt1, "ise_toy_gaussian_reference.pdf")

plt2 = plot(alphas[:], mean(mean_ref.^2, dims = 1)[:], label = "Ref", 
    lw = 3, linestyle = :dash, color = :gray)
plot!(plt2, alphas[:], mean(mean_noref.^2, dims = 1)[:], label = "No ref", 
    legendfontsize = 15, legend=:topleft, lw = 3, linestyle = :dot, color = :black)
# savefig(plt2, "mean_toy_gaussian_reference.pdf")

plt3 = plot(alphas[:], mean((var_ref .- 1).^2, dims = 1)[:], label = "Ref", 
    lw = 3, linestyle = :dash, color = :gray)
plot!(plt3, alphas[:], mean((var_noref .- 1).^2, dims = 1)[:], label = "No ref", 
    legendfontsize = 15, legend=:topleft, lw = 3, linestyle = :dot, color = :black)
# savefig(plt3, "variance_toy_gaussian_reference.pdf")
