

function [L] = LoanPayoff(F, f, ival, rho, sig, T)

  %F = loan face value (either a scalar or a matrix the same size as f) 
  %f = log asset value factor realizations 
  %ival = initial log asset value 
  %rho = correlation of asset values
  %sig = volatility of log asset values 
  %T = loan maturity 
  %r = log risk free rate 
  
  EA = exp(f+ival+0.5*(1-rho)*T*sig^2);     %expected asset value conditional on common factor
  s = sig*sqrt(T)*sqrt(1-rho);
  a = (log(F)-f-ival)/s;  
  L = EA.*(1-normcdf(s-a)) + normcdf(-a).*F;  %loan portfolio payoff at maturity 
          
  
  
  
  