function [err] = ModSingleCohortSolution(b, param, N, Nsim2, E, sigE)

fs = b(1); 
param(3) = b(2); 

%{
skip = 0; 
sig = b(2);
esig = 0; 
if sig < 0
    sig = 0; 
    esig = 9999999; 
    skip = 1; 
end
eA = 0; 
if fs < 0
    fs = 0; 
    eA = 9999999; 
    skip = 1; 
end
%}

[Lt, Bt, Et, LH, BH, EH, sigEt, mFt, def, mdef, face, FH, Gt, mu, F, sigLt] = ModSingleCohortComputation(fs, param, N, Nsim2);  
err = [E - Et; sigE-sigEt]; 


