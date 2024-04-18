module wgf2kind;

using Distributions;
using Statistics;
using LinearAlgebra;


export wgf2kind_asset_pricing
export wgf2kind_toy_gaussian
export wgf2kind_toy_gaussian_noreference
export wgf2kind_gpssm
export wgf2kind_kl_squared_exp
export wgf2kind_kl_exp


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
function wgf2kind_toy_gaussian_noreference(N, dt, Niter, alpha_param, x0, m0, sigma0, lambda)
    # parameters
    beta = 0.5;
    varK = 1-exp(-2*beta);
    # define kernel and forcing
    K(x, y) = pdf.(Normal(x*exp(-beta), sqrt(varK)), y);
    phi(x) = (1-lambda)*pdf.(Normal(0, 1), x);

    # initialise a matrix x storing the particles
    x_target = zeros(Niter, N);
    x_target[1, :] = x0;
    x_diffuse = zeros(Niter, N);
    x_diffuse[1, :] = x0;
    x_concentrated = zeros(Niter, N);
    x_concentrated[1, :] = x0;
    x_noref = zeros(Niter, N);
    x_noref[1, :] = x0;

    for n=1:(Niter-1)
        outside_integral = zeros(N, 1);
        mean_k = zeros(N, 1);
        outside_integral_concentrated = zeros(N, 1);
        mean_k_concentrated = zeros(N, 1);
        outside_integral_diffuse = zeros(N, 1);
        mean_k_diffuse = zeros(N, 1);
        outside_integral_noref = zeros(N, 1);
        mean_k_noref = zeros(N, 1);
        for i=1:N
            mean_k[i] = mean(K.(x_target[n,i], x_target[n,:]));
            mean_k_diffuse[i] = mean(K.(x_diffuse[n,i], x_diffuse[n,:]));
            mean_k_concentrated[i] = mean(K.(x_concentrated[n,i], x_concentrated[n,:]));
            mean_k_noref[i] = mean(K.(x_noref[n,i], x_noref[n,:]));
            outside_integral[i] = -alpha_param*(x_target[n,i] - m0)./(sigma0^2) +
                (lambda*mean(K.(x_target[n,i], x_target[n,:]).*(x_target[n,:] .- x_target[n,i]*exp(-beta)))*exp(-beta)/varK - phi(x_target[n,i])*x_target[n,i])/(lambda*mean_k[i]+phi(x_target[n,i]));
            outside_integral_diffuse[i] = -alpha_param*(x_diffuse[n,i] - m0)./(2^2) +
                (lambda*mean(K.(x_diffuse[n,i], x_diffuse[n,:]).*(x_diffuse[n,:] .- x_diffuse[n,i]*exp(-beta)))*exp(-beta)/varK - phi(x_diffuse[n,i])*x_diffuse[n,i])/(lambda*mean_k_diffuse[i]+phi(x_diffuse[n,i]));
            outside_integral_concentrated[i] = -alpha_param*(x_concentrated[n,i] - m0)./(0.1^2) +
                (lambda*mean(K.(x_concentrated[n,i], x_concentrated[n,:]).*(x_concentrated[n,:] .- x_concentrated[n,i]*exp(-beta)))*exp(-beta)/varK - phi(x_concentrated[n,i])*x_concentrated[n,i])/(lambda*mean_k_concentrated[i]+phi(x_concentrated[n,i]));
            outside_integral_noref[i] = (lambda*mean(K.(x_noref[n,i], x_noref[n,:]).*(x_noref[n,:] .- x_noref[n,i]*exp(-beta)))*exp(-beta)/varK - phi(x_noref[n,i])*x_noref[n,i])/(lambda*mean_k_noref[i]+phi(x_noref[n,i]));
        end
    
        inside_integral = zeros(N, 1);
        inside_integral_diffuse = zeros(N, 1);
        inside_integral_concentrated = zeros(N, 1);
        inside_integral_noref = zeros(N, 1);
        for i=1:N
            integrand = (-lambda*K.(x_target[n,:], x_target[n, i]).*(x_target[n,i] .- x_target[n,:]*exp(-beta))/varK)./
            (lambda*mean_k .+ phi(x_target[n,:]));
            inside_integral[i] = mean(integrand);
            integrand_diffuse = (-lambda*K.(x_diffuse[n,:], x_diffuse[n, i]).*(x_diffuse[n,i] .- x_diffuse[n,:]*exp(-beta))/varK)./
            (lambda*mean_k .+ phi(x_diffuse[n,:]));
            inside_integral_diffuse[i] = mean(integrand_diffuse);
            integrand_concentrated = (-lambda*K.(x_concentrated[n,:], x_concentrated[n, i]).*(x_concentrated[n,i] .- x_concentrated[n,:]*exp(-beta))/varK)./
            (lambda*mean_k .+ phi(x_diffuse[n,:]));
            inside_integral_concentrated[i] = mean(integrand_concentrated);
            integrand_noref = (-lambda*K.(x_noref[n,:], x_noref[n, i]).*(x_noref[n,i] .- x_noref[n,:]*exp(-beta))/varK)./
            (lambda*mean_k_noref .+ phi(x_noref[n,:]));
            inside_integral_noref[i] = mean(integrand_noref);
        end
    
        # gradient and drift
        drift = inside_integral + outside_integral;
        drift_diffuse = inside_integral_diffuse + outside_integral_diffuse;
        drift_concentrated = inside_integral_concentrated + outside_integral_concentrated;
        drift_noref = inside_integral_noref + outside_integral_noref;
        # update locations
        Z = randn(N, 1);
        x_target[n+1, :] = x_target[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*Z;
        x_diffuse[n+1, :] = x_diffuse[n, :] .+ dt * drift_diffuse .+ sqrt(2*(1+alpha_param)*dt)*Z;
        x_concentrated[n+1, :] = x_concentrated[n, :] .+ dt * drift_concentrated .+ sqrt(2*(1+alpha_param)*dt)*Z;
        x_noref[n+1, :] = x_noref[n, :] .+ dt * drift_noref .+ sqrt(2*(1+alpha_param)*dt)*Z;
    end
    return x_target, x_noref, x_concentrated, x_diffuse
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



#= WGF for KL expansion of squared exponential
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
function wgf2kind_kl_squared_exp(N, dt, Niter, alpha_param, x0, m0, sigma0, lambda)
    # parameters
    K(x, y) = exp.(-(y .- x).^2);

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
                (lambda*mean(K.(x[n,i], x[n,:]).*(x[n,:] .- x[n,i]))*2)/(lambda*mean_k[i]);
        end
    
        inside_integral = zeros(N, 1);
        for i=1:N
            integrand = (-lambda*K.(x[n,:], x[n, i]).*(x[n,i] .- x[n,:])*2)./
            (lambda*mean_k);
            inside_integral[i] = mean(integrand);
        end
    
        # gradient and drift
        drift = inside_integral + outside_integral;
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift .+ sqrt(2*(1+alpha_param)*dt)*randn(N, 1);
    end
    return x
end



#= WGF for KL expansion of exponential
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
function wgf2kind_kl_exp(N, dt, Niter, alpha_param, x0, m0, sigma0, lambda)
    # parameters
    K(x, y) = exp.(-abs.(y .- x));

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
                (lambda*mean(K.(x[n,i], x[n,:]).*sign.(x[n,:] .- x[n,i])))/(lambda*mean_k[i]);
        end
    
        inside_integral = zeros(N, 1);
        for i=1:N
            integrand = (-lambda*K.(x[n,:], x[n, i]).*sign.(x[n,i] .- x[n,:]))./
            (lambda*mean_k);
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