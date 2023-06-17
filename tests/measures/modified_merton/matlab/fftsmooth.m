function [xout] = fftsmooth(x,w)

%NOTE: so far this function works only for even number of obs and even
%number of kernel window elements 
 
%x = column vector of inputs to be smoothed
%w = window size (in # of obs) 

n = size(x,1); 
xpad = [zeros(n,1); x]; 
k = [zeros(n-w/2,1); ones(w,1)/w; zeros(n-w/2,1)]; 
Fx = fft(xpad); 
Fk = fft(k); 
Fxk = Fx .* Fk; 
xk = ifft(Fxk); 

%extrapolate linearly at edges
dl = (xk(w/2+2)-xk(w/2+1)); %/(w-1);
lex = xk(w/2+1) - ([w/2:-1:1])'*dl; 
ul = (xk(n-w/2)-xk(n-w/2-1)); %/(w-1);
uex = xk(n-w/2) + ([1:w/2])'*ul; 

xout = [lex; xk(w/2+1:n-w/2); uex]; 
