# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using RCall;
@rimport ks as rks;


# set seed
Random.seed!(1234);
# parameters
beta = 0.5;
varK = 1-exp(-2*beta);
alpha_param = 0.0000;
m0 = 0;
sigma0 = 1;
lambda = 1/3;
# dt and number of iterations
dt = 1e-03;
Niter = 1000;
Nparticles = 100;
x = zeros(Niter, Nparticles);
# initial distribution is given as input:
x0 = rand(Normal.(0, 0.1), Nparticles);
x[1, :] = x0;
K(x, y) = pdf.(Normal(x*exp(-beta), sqrt(varK)), y);
phi(x) = (1-lambda)*pdf.(Normal(0, 1), x);

# functional approximation
function psiWGF(piSample, lambda, alpha_param, m0, sigma0)
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
EWGF[1] = psiWGF(x[1, :], lambda, alpha_param, m0, sigma0);
runtime = @elapsed begin
for n=1:(Niter-1)
    print("$n\n")
   
    outside_integral = zeros(Nparticles, 1);
    mean_k = zeros(Nparticles, 1);
    for i=1:Nparticles
        mean_k[i] = mean(K.(x[n,i], x[n,:]));
        outside_integral[i] = -alpha_param*(x[n,i] - m0)./(sigma0^2) +
            (lambda*mean(K.(x[n,i], x[n,:]).*(x[n,:] .- x[n,i]*exp(-beta)))*exp(-beta)/varK - phi(x[n,i])*x[n,i])/(lambda*mean_k[i]+phi(x[n,i]));
    end

    inside_integral = zeros(Nparticles, 1);
    for i=1:Nparticles
        integrand = (-lambda*K.(x[n,:], x[n, i]).*(x[n,i] .- x[n,:]*exp(-beta))/varK)./
        (lambda*mean_k .+ phi(x[n,:]));
        inside_integral[i] = mean(integrand);
    end

    # gradient and drift
    drift = inside_integral + outside_integral;
    # update locations
    x[n+1, :] = x[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*randn(Nparticles, 1);
    EWGF[n+1] = psiWGF(x[n+1, :], lambda, alpha_param, m0, sigma0);
end
end

x_solution = range(-3, 3, length = 100);
y_solution = pdf.(Normal.(0, 1), x_solution);
KDE = rks.kde(x = x[Niter,:], var"eval.points" = x_solution);
KDE =  abs.(rcopy(KDE[3]));
KDE0 = rks.kde(x = x0, var"eval.points" = x_solution);
KDE0 =  abs.(rcopy(KDE0[3]));
p1 = plot(x_solution, y_solution)
plot!(x_solution, KDE)
p2 = plot(1:Niter, EWGF)
plot(p1, p2, layout = (1, 2))
