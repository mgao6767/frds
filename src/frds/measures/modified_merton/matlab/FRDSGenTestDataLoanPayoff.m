function [F, f1j, ival, rho, sig, T] = FRDSGenTestDataLoanPayoff()
fs = [-0.8:0.05:0.8]'/(0.2*sqrt(0.5)*sqrt(10)); %range for dW_0 shocks

N = 10;               %number of loan cohorts
Nsim2 = 1000;        %number of simulated factor realization paths (10,000 works well)

r = 0.01;      %risk-free rate;
d = 0.005;     %depreciation rate of borrower assets
y = 0.002;     %bank payout rate as a percent of face value of loan portfolio
T = 10;        %loan maturity
bookD = 0.63;
H = 5;         %bank debt maturity
D = bookD*exp(r*H)*mean(exp(r*([1:T])'));   %face value of bank debt: as if bookD issued at start of each cohort, so it grows just like RN asset value
rho = 0.5;
sig = 0.2;
ltv = 0.66;  %intial ltv
g = 0.5;     %prob of govt bailout (only used for bailout valuation analysis)
j = 3;
bookF = 0.6+j*0.02; %initial loan amount = book value of assets (w/ coupon-bearing loans the initial amount would be book value)

param = [r; T; bookF; H; D; rho; ltv; sig; d; y; g];

% inside ModMertonComputation.m
r = param(1);      %log risk free rate
T = param(2);      %original maturity of bank loans
bookF = param(3);  %cash amount of loan issued = book value for a coupon-bearing loan issued at par
H = param(4);      %bank debt maturity
D = param(5);      %face value of bank debt
rho = param(6);    %borrower asset value correlation
ltv = param(7);    %initial LTV
sig = param(8);    %borrower asset value volatility
d = param(9);      %depreciation rate of borrower assets
y = param(10);     %bank payout rate

%optional: calculate value of govt guarantee, with prob g, govt absorbs
%Loss given default (bank and equity values not adjusted for presence of
%govt guarantee)
if size(param,1) > 10
    g = param(11);
else
    g = 0;
end

%optional: provide simulated factor shocks (for faster repeated computations)
%if not provided as input, then generate here
if nargin < 5
    rng(1,'twister')
    %   w = normrnd(0,1,[Nsim2, 3*N]);
    % Mingze's note
    % This is to produce the same random normal values as numpy
    % For unknown reasons, same seed doesn't guarantee same random NORMAL values
    % in Matlab and Numpy given the same MT19937 rng.
    % This line below is a workaround.
    w = norminv(rand(Nsim2, 3*N),0,1);
end

ival = log(bookF)-log(ltv);  %initial log asset value of borrower at origination
sigf = sqrt(rho)*sig;
HN = H*(N/T);                %maturity in N time;
szfs = size(fs,1);
fs = [fs; fs; fs];           %use second and third block for numerical derivative
Nsim1 = size(fs,1);

%remaining maturity of first loans at t
rmat(1,:,1) = [0:1:N-1];
rmat = repmat(rmat, [Nsim2,1, Nsim1]);
ind1 = (rmat >= HN);
ind2 = (rmat < HN);

%fractional accumulated loan life time at t+H
aHmat(1,:,1) = [HN:-1:0 N-1:-1:HN+1]/N;
aHmat = repmat(aHmat, [Nsim2,1, Nsim1]);

%fractional accumulated loan life time at t
atmat(1,:,1) = (N-[0:1:N-1])/N;
atmat = repmat(atmat, [Nsim2,1, Nsim1]);

%Euler discretization for log value has a jensen's term
%which depends on total, not just factor volatility,
%since f here is average log asset value of borrowers, the idiosyncratic
%shock averaged out, but to have expected return (gross of depreciation)
%for each individual borrower equal to exp(r), the drift needs to adjust
%for total volatility.
%Further below:
%To get average level of asset value, we take E[exp(f + 0.5*idiovar)]
%because only volatility from factors generates variation in our
%simulations (which then raises E[exp(.)] by averaging convex function,
%this 0.5*idiovar part combined with the total volatility drift
%adjustment here then only leaves the factor volatility drift adjustment
f = [zeros(Nsim2,1) cumsum( (r-d-0.5*sig^2)*(T/N) + sigf*sqrt(T/N)*w , 2)];

%fw is what we need to remove from f (until t) to let LEVEL of aggregate
%borrower asset value grow at expected path exp(r-d)
%(after we also add 0.5*idiovar further below)
fw = [zeros(Nsim2,1) cumsum( -0.5*rho*sig^2*(T/N) + sigf*sqrt(T/N)*w , 2)];

%factor realizations at relevant points for staggered loan cohorts (#sim x cohort)
% t = Time we do valuation
%first loan of first cohort starts at 0 and matures at t = N, i.e. accumulated factor shocks
%are those in 1, 2, ..., N. First loan of N-th cohort starts at N-1 and
%matures at 2*N-1
%second loan of first cohort starts at N and accumulated factor shocks are those
%in N+1, ..., 2*N. Second loan of N-th cohort starts at 2*N-1 and matures
%at 3*N-1
%Note that first element of f is zero, so f(1) is time 0 and f(N+1) is time t
xf1 = f(:,N+1) - f(:,1);                  %from start of first loan to maturity of first loan, first cohort only
f0w = repmat(fw(:,N+1),1,N) - fw(:,1:N);  %factor shocks from start of first loan shocks until t
f1 = f(:,N+1:2*N) - f0w - f(:,1:N);       %from start of first loan to maturity of first loan w/ shocks until t removed
f2 = f(:,2*N+1:3*N) - f(:,N+1:2*N);       %from start of second loan to maturity of second loan

%add fs shocks after loan origination
%dimension of f1tj, f2tj becomes (#sim  x cohort x #fs-values)
%do not apply maturity weighting to df increments for numerical
%derivatives used for volatility calculation
fsa(1,1,:) = fs*sigf*sqrt(T);
dstep = 10;     %1/dstep of SD step to evaluate numerical derivative
df = sigf/dstep;
fsa = repmat(fsa, [Nsim2 N 1]).*atmat ...
    + df*cat(3,zeros(Nsim2,N,szfs),ones(Nsim2,N,szfs),-ones(Nsim2,N,szfs));
f1j = repmat(f1,[1,1,Nsim1]) + fsa;
f2j = repmat(f2,[1,1,Nsim1]);     %fs shock not here because they occurred before origination of second loan

%solve for promised yield on loan (a fixed point):
%do not remove factor shocks until t in this calculation
options = optimset('TolFun',1.0e-12,'MaxIter',100000, 'TolX',1.0e-12, ...
    'MaxFunEvals',100000, 'Display', 'off');
initmu = r+0.01;

[muout,fval,exitflag,output] = fsolve(@(mu) FindFaceValueIndiv(mu, bookF*exp(mu*T), ival, sig, T, r, d), initmu, options); 
if exitflag < 1
  disp('Face value not found') 
end 
mu = muout;              %output is promised total yield  

F = bookF*exp(mu*T);    %loan face value; 
 