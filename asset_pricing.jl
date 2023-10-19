# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using RCall;
using SpecialFunctions;
@rimport ks as rks;


# set seed
Random.seed!(1234);
# parameters
lambda = 0.85;
a = 0.05;
b = 0.9;
varK = 100;
alpha_param = 0.01;
m0 = 0.5;
sigma0 = 0.1;
# dt and number of iterations
dt = 1e-03;
Niter = 200;
Nparticles = 1000;
x = zeros(Niter, Nparticles);
# initial distribution is given as input:
x0 = rand(Normal.(0.5, 0.1), Nparticles);
x[1, :] = x0;
function K(x, y)
    d0 = Normal(a*x + b, sqrt(varK))
    truncated_d0 = truncated(d0, 0, 1)
    return(pdf.(truncated_d0, y))
end
phi(x) = (exp(x^2)-1) * (x <= 1) * (x >= 0);
phi_gradient(x) = 2*x*exp(x^2) * (x <= 1) * (x >= 0);

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
        gradientK1 = ((x[n, :] .- a*x[n, i] .- b)/varK .+ 
        (pdf(Normal(0, sqrt(varK)), 1-a*x[n, i]-b) - pdf(Normal(0, sqrt(varK)), -a*x[n, i]-b))/
        (cdf(Normal(0, 1), (1-a*x[n, i]-b)/sqrt(varK))-cdf(Normal(0, 1), (-a*x[n, i]-b)/sqrt(varK))));
        gradientK1 = a * gradientK1 .* K(x[n, i], x[n, :]);
        outside_integral[i] = -alpha_param*(x[n,i] - m0)./(sigma0^2) +
            (mean(gradientK1) + phi_gradient.(x[n, i]))/(lambda*mean_k[i]+phi(x[n,i]));
    end

    inside_integral = zeros(Nparticles, 1);
    for i=1:Nparticles
        integrand =  lambda*K.(x[n,:], x[n,i]).*(x[n, i] .- a*x[n, :] .- b)./(varK*(lambda*mean_k .+ phi.(x[n,:])));
        inside_integral[i] = mean(integrand);
    end

    # gradient and drift
    drift = inside_integral + outside_integral;
    # update locations
    x[n+1, :] = x[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*randn(Nparticles, 1);
    EWGF[n+1] = psiWGF(x[n+1, :], lambda, alpha_param, m0, sigma0);
end
end

KDE = rks.kde(x = x[Niter,:]);
x_solution =  rcopy(KDE[2]);
KDE =  abs.(rcopy(KDE[3]));
p1 = plot(x_solution, KDE)

x_alt = range(0, 1, length = 100);
sample_alt = (x[Niter, :] .<= 1) .* (x[Niter, :] .>= 0)
y_solution = zeros(length(x_alt));
for i=1:length(x_alt)
    y_solution[i] = phi(x_alt[i]) + lambda*mean(K(x_alt[i], x[Niter, :]))
end
plot(x_alt, y_solution)
histogram!(x[Niter, sample_alt], normalize=:pdf)


p2 = plot(1:Niter, EWGF)