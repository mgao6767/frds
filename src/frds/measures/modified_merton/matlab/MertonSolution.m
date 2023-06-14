

function [err] = MertonSolution(b, E, D, r, d, T, sigE)

A = b(1); 
sig = b(2);
esig = 0; 
if sig < 0
    sig = 0; 
    esig = 9999999; 
end
eA = 0; 
if A < 0
    A = 0; 
    eA = 9999999; 
end

[C, P] = blsprice(A , D, r, T, sig, d);

%add PV(dividends) to get cum dividend value
PVd = A*(1-exp(-d*T)); 
C = C + PVd; 

d1 = (log(A)-log(D)+(r-d+sig^2/2)*T)/(sig*sqrt(T)); 
v = ( exp(-d*T)*normcdf(d1) + (1-exp(-d*T)) )*(A/E)*sig; 

%v = ((A)/(C))*sig;   %A includes PV(div), but B-S derivative prices don't 
%Merton asset volatility too high 

err = [E - C; sigE-v] + esig*b(2)^2 + eA*b(1)^2; 

