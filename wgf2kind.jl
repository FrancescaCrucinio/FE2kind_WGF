module wgf2kind;

using Distributions;
using Statistics;
using LinearAlgebra;


export wgf2kind_asset_pricing
export wgf2kind_toy_gaussian
export wgf2kind_gpssm

#= WGF for asset pricing
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha_param' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
=#
function wgf2kind_asset_pricing(N, dt, Niter, alpha_param, x0, m0, sigma0)
    # parameters
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
    phi(x) = (exp(x^2)-1) * (x <= 1) * (x >= 0);
    phi_gradient(x) = 2*x*exp(x^2) * (x <= 1) * (x >= 0);

    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
     # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        outside_integral = zeros(N, 1);
        mean_k = zeros(N, 1);
        for i=1:N
            mean_k[i] = mean(K.(x[n,i], x[n,:]));
            gradientK1 = ((x[n, :] .- a*x[n, i] .- b)/varK .+ 
            (pdf(Normal(0, sqrt(varK)), 1-a*x[n, i]-b) - pdf(Normal(0, sqrt(varK)), -a*x[n, i]-b))/
            (cdf(Normal(0, 1), (1-a*x[n, i]-b)/sqrt(varK))-cdf(Normal(0, 1), (-a*x[n, i]-b)/sqrt(varK))));
            gradientK1 = a * gradientK1 .* K(x[n, i], x[n, :]);
            outside_integral[i] = -alpha_param*(x[n,i] - m0)./(sigma0^2) +
                (mean(gradientK1) + phi_gradient.(x[n, i]))/(lambda*mean_k[i]+phi(x[n,i]));
        end
    
        inside_integral = zeros(N, 1);
        for i=1:N
            integrand =  lambda*K.(x[n,:], x[n,i]).*(x[n, i] .- a*x[n, :] .- b)./(varK*(lambda*mean_k .+ phi.(x[n,:])));
            inside_integral[i] = mean(integrand);
        end
    
        # gradient and drift
        drift = inside_integral + outside_integral;
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*randn(N, 1);
    end
    return x
end



#= WGF for toy Gaussian model
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha_param' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'lambda' integral equation parameter
=#
function wgf2kind_toy_gaussian(N, dt, Niter, alpha_param, x0, m0, sigma0, lambda)
    # parameters
    beta = 0.5;
    varK = 1-exp(-2*beta);
    # define kernel and forcing
    K(x, y) = pdf.(Normal(x*exp(-beta), sqrt(varK)), y);
    phi(x) = (1-lambda)*pdf.(Normal(0, 1), x);

    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
     # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        outside_integral = zeros(N, 1);
        mean_k = zeros(N, 1);
        for i=1:N
            mean_k[i] = mean(K.(x[n,i], x[n,:]));
            outside_integral[i] = -alpha_param*(x[n,i] - m0)./(sigma0^2) +
                (lambda*mean(K.(x[n,i], x[n,:]).*(x[n,:] .- x[n,i]*exp(-beta)))*exp(-beta)/varK - phi(x[n,i])*x[n,i])/(lambda*mean_k[i]+phi(x[n,i]));
        end
    
        inside_integral = zeros(N, 1);
        for i=1:N
            integrand = (-lambda*K.(x[n,:], x[n, i]).*(x[n,i] .- x[n,:]*exp(-beta))/varK)./
            (lambda*mean_k .+ phi(x[n,:]));
            inside_integral[i] = mean(integrand);
        end
    
        # gradient and drift
        drift = inside_integral + outside_integral;
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*randn(N, 1);
    end
    return x
end




#= WGF for GP-SSM
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha_param' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'xs, zs' training data
=#
function wgf2kind_gpssm(N, dt, Niter, alpha_param, x0, m0, sigma0, xs, zs)
    # covariance operator of GP
    m = length(xs);
    cov_op(x, y) = exp(-(x - y)^2/(2*3.59^2))*4.21^2;
    matrixK = zeros(m, m);
    for i=1:m
        matrixK[i, :] = cov_op.(xs[i], xs);
    end
    matrixH = inv(matrixK + I(m));
    # define kernel and forcing
    function K_mean(y) 
        return(transpose(cov_op.(y, xs))*matrixH*zs)
    end
    function K_mean_derivative(y) 
        cov_op_derivative = cov_op.(y, xs) .* (xs .- y)/3.59^2;
        return(transpose(cov_op_derivative)*matrixH*zs)
    end
    function K_variance(y) 
        return(cov_op(y, y) - transpose(cov_op.(y, xs))*matrixH*cov_op.(y, xs))
    end 
    function K_variance_derivative(y) 
        cov_op_derivative = cov_op.(y, xs) .* (xs .- y)/3.59^2;
        return(-2*transpose(cov_op_derivative)*matrixH)
    end 
    function K(x, y)
        transition_mean = K_mean(y);
        transition_variance = K_variance(y);
        return(pdf.(Normal(transition_mean, transition_variance), x))
    end

    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
     # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        outside_integral = zeros(N, 1);
        mean_k = zeros(N, 1);
        for i=1:N
            mean_k[i] = mean(K.(x[n,i], x[n,:]));
            outside_integral[i] = -alpha_param*(x[n,i] - m0)./(sigma0^2) +
                mean(K.(x[n,i], x[n,:]).*(K_mean.(x[n,:]) .- x[n,i])./K_variance.(x[n,:]))/mean_k[i];
        end
    
        inside_integral = zeros(N, 1);
        for i=1:N
            integrand = (K.(x[n,:], x[n,i]).*(x[n,:] .- K_mean(x[n,i]))/K_variance(x[n,i])^3)./mean_k;
            integrand = integrand .* (K_mean_derivative(x[n,i])*K_variance(x[n,i]) .+
                 (x[n,:] .- K_mean_derivative(x[n,i])).*K_variance_derivative(x[n,i]));
            inside_integral[i] = mean(integrand);
        end
    
        # gradient and drift
        drift = inside_integral + outside_integral;
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*randn(N, 1);
    end
    return x
end


end