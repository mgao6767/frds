function [err, newmu, L] = FindFaceValueIndiv(mu, F, ival, sig, T, r, d)

  [C, P] = blsprice(exp(ival), F, r, T, sig, d);
 
  L = F*exp(-r*T)-P;  %to do with call we would also need to subtract PV(depreciation)
  newmu = (1/T)*log(F/L); 
  
  err = mu-newmu;   
  