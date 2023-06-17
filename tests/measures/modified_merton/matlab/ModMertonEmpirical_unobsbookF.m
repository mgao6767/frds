
% Output: computes modified Merton model risk-neutral default probabilities (mdef) 
% Input: 
% - file with empirical Equity values and (instant.) equity S.D. for set of banks: 
%   variables sE and E
% - file with debt and equity values, equity vol, ... as function of borrower
%   asset vol, borrower asset value, ... computed in ModMertonComputation.m


clear all;
close all; 
clc;

%working directory here.
direc = ['/Users/amiyatos/Dropbox/BankCreditRisk/PublicCode/'];

compute =0;   %set to 1 to run  ModMertonComputation.m, otherwise it uses saved file (ValueSurface.mat) 
              %need to run the code with compute=1 first to get the valuesurface.
N = 10;             %number of loan cohorts   
Nsim2 = 5000;       %number of simulated factor realization paths 

d = 0.005;   %depreciation rate of borrower assets
y = 0.002;   %bank payout rate as a percent (p.a) of asset value 
T = 10;      %loan maturity 
H = 5;       %bank debt maturity
bookD = 1;   %current book value of bank debt (normalized to 1, normalize all other balance sheet variables accordingly)   
rho = 0.5;   %share of factor variance in total borrower asset variance: we can't identify and need to assume a value (with robustness checks)  
ltv = 0.66;  %initial LTV 

% Let F = face value of (zero-coupon loans) 
%     D = face value of bank's (zero-coupon) debt
%
% bookF = book value of assets 
%       = in empirical data, approx, under ass. that issued at par w/ coupon, this is the face value of the loan 
%           (w/o loss provisions subtracted), i.e., the initial amount transferred to the borrower
%    with zero-coupon loans in our model we should then set (done in ModMertonComputation.m) 
%        F = bookF * exp(mu*T) 
%    since banks do not have zero coupon loans in practice, their book
%    value of loans does not grow with maturity 
%    but in our case, the face value needs to incorporate the accumulated
%    and compounded interest payments: hence bookF * exp(mu*T) 
%    After scaling BS-variables by book value of bank debt, we get 
%        bookF = book value asset/book value debt
%        F = (book value asset/book value debt) * exp(mu*T)
%    with an initial loan to value ratio of ltv, we then start with inital
%    collateral of 
%        F * exp(-mu * T)/ltv = bookF/ltv 
% 
% book value of debt = D * exp(-r * H) 
%    after scaling BS-variables by book value of debt, we compute
%      D = exp(r*H)  (implemented in ModMertonCreateLookup.m)
% 


%Load empirical data
%File qtr_bankdd.csv in the example below contains the dataset with permco,
%year, month, market value of equity (E) scaled by book debt, equity
%volatility (sE), risk free rate (r). Please change your input variables
%accordingly.
%test_inputdata is an input test dataset containing E, sE and r. Alongwith
%permco, year , month identifier.

filename=[direc,'test_inputdata.csv'];
qtr_bankdd=csvread(filename,1,0);

E=qtr_bankdd(:,1);
permco=qtr_bankdd(:,2);
year=qtr_bankdd(:,3);
month=qtr_bankdd(:,4);
r=qtr_bankdd(:,5);
sE=qtr_bankdd(:,6);

% run ModMertonComputation.m
% the grid surface should be wide enough to cover empirically relevant
% range.
[xfs,xsig,xr,xF] = ndgrid( -8:0.05:8, 0.15:0.05:0.25, 0:0.005:0.1, 0.5:0.05:2.5); 


if compute == 1
   
   [xLt, xBt, xEt, xFt, xmdef, xsigEt] = ModMertonCreateLookup(d,y,T,H,bookD,rho,ltv,xfs,xr,xF,xsig,N,Nsim2); 
   
   param = [T; H; bookD; rho; d; y; ltv];
   
   save([direc,'ValueSurface'], 'xLt', 'xBt', 'xEt', 'xFt', 'xmdef', 'xsigEt', 'xfs', 'xsig', 'xr', 'xF', ...
          'N', 'Nsim2', 'param');

end      
      
load([direc,'ValueSurface']) 

dataN = size(E,1); 
rs = size(xr,3);                 %number of interest rate obs 
minr = xr(1,1,1,1); 
maxr = xr(1,1,rs,1); 
vol=ones(dataN,1)*0.2;          %feed asset volatility value here.

for a = 1:rs

   sigEt = squeeze(xsigEt(:,:,a,:)); 
   Et  = squeeze(xEt(:,:,a,:));
   Bt  = squeeze(xBt(:,:,a,:));
   mdef  = squeeze(xmdef(:,:,a,:));
   Lt  = squeeze(xLt(:,:,a,:));
   sig  = squeeze(xsig(:,:,a,:));
   fs  = squeeze(xfs(:,:,a,:));
   bookF = squeeze(xF(:,:,a,:));
   
   %eliminate NaNs and Inf in equity volatility (with linear interpol this
   %does not affect other observations) 
   ind = isnan(sigEt); 
   sigEt(ind) = 99; 
   ind = isinf(sigEt); 
   sigEt(ind) = 99; 

   %perform inversion by 3D-interpolating linearly as a function of Et, sigEt, F
   %for a given interest rate
   %use "macro" (Matlab structures can't seem to handle
   %ScatteredInterpolant Elements)
   eval(sprintf('yfs%d = scatteredInterpolant(Et(:),sigEt(:),sig(:),fs(:))', a));
   eval(sprintf('yLt%d = scatteredInterpolant(Et(:),sigEt(:),sig(:),Lt(:))', a));
   eval(sprintf('yBt%d = scatteredInterpolant(Et(:),sigEt(:),sig(:),Bt(:))', a));
   eval(sprintf('ybookF%d = scatteredInterpolant(Et(:),sigEt(:),sig(:),bookF(:))', a));
   eval(sprintf('ymdef%d = scatteredInterpolant(Et(:),sigEt(:),sig(:),mdef(:))', a));

end 


%extract interpolated values corresponding to empirical data points
%for each interest rate on the grid 
Lr = zeros(dataN,rs); 
Br = zeros(dataN,rs); 
mdefr = zeros(dataN,rs); 
sigr = zeros(dataN,rs); 
bookFr = zeros(dataN,rs);
fsr = zeros(dataN,rs); 
for a = 1:rs
   eval(sprintf('Lr(:,%d) = yLt%d(E,sE,vol);', a, a));
   eval(sprintf('Br(:,%d) = yBt%d(E,sE,vol);', a, a));
   eval(sprintf('mdefr(:,%d) = ymdef%d(E,sE,vol);', a, a));
   eval(sprintf('bookFr(:,%d) = ybookF%d(E,sE,vol);', a, a));
   eval(sprintf('fsr(:,%d) = yfs%d(E,sE,vol);', a, a));
end

%find closest interest rates on grid
rstep = (maxr-minr)/(rs-1); 
rmat = repmat(r,1,rs); 
rgrid = repmat(minr:rstep:maxr,dataN,1); 
dr = (rgrid-rmat)/rstep; 
Wl = 1+dr; 
Wu = 1-dr; 
ind1 = (dr <= -1); 
ind2 = (dr >= 1); 
indl = (dr < 0); 
indu = (dr > 0); 
W = zeros(size(rmat)); 
W(indl) = Wl(indl); 
W(indu) = Wu(indu); 
W(ind1) = 0; 
W(ind2) = 0; 

L = sum(Lr.*W,2); 
B = sum(Br.*W,2); 
mdef = sum(mdefr.*W,2); 
bookF = sum(bookFr.*W,2); 
fs = sum(fsr.*W,2); 


%write relevant output to mdef file.
save([direc,'mdef_output'], 'L', 'B', 'mdef', 'fs', 'E', 'bookF', 'r', 'permco', 'year', 'month', 'sE','vol');

    