

function [xLt, xBt, xEt, xFt, xmdef, xsigEt] = ModMertonCreateLookup(d,y,T,H,bookD,rho,ltv,xfs,xr,xF,xsig,N,Nsim2) 


rng(1,'twister')
w = normrnd(0,1,[Nsim2, 3*N]);
     
J = size(xsig,2); 
K = size(xr,3); 
Q = size(xF,4); 
G = size(xfs,1); 

fs = xfs(:,1,1,1); 

xLt = zeros(G,J,K,Q); 
xBt = zeros(G,J,K,Q);  
xEt = zeros(G,J,K,Q); 
xFt = zeros(G,J,K,Q); 
xmdef = zeros(G,J,K,Q); 
xsigEt = zeros(G,J,K,Q); 

for j = 1:J
    
    for k = 1:K
        
        for q = 1:Q 
           j
           k
           q
          
           param = [xr(1,j,k,q); T; xF(1,j,k,q); H; bookD*exp(xr(1,j,k,q)*H); rho; ltv; xsig(1,j,k,q); d; y];  %get face value of debt from current book value of debt
        
           [Lt, Bt, Et, ~, ~, ~, sigEt, mFt, ~, mdef] = ModMertonComputation(fs, param, N, Nsim2, w); 
        
           xLt(:,j,k,q) = Lt; 
           xBt(:,j,k,q) = Bt; 
           xEt(:,j,k,q) = Et; 
           xFt(:,j,k,q) = mFt; 
           xmdef(:,j,k,q) = mdef; 
           xsigEt(:,j,k,q) = sigEt; 
           
        end      
    end  
end 


