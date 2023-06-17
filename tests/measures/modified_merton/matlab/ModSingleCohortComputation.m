
function [Lt, Bt, Et, LH, BH, EH, sigEt, mFt, def, mdef, face, FH, Gt, mu, F, sigLt] = ModSingleCohortComputation(fs, param, N, Nsim2, w) 

  
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
  
  %With single borrower cohort, N just determines discretization horizon of
  %shocks, not the number of cohorts 
  
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
      w = normrnd(0,1,[Nsim2, N]);
  end
   
  ival = log(bookF)-log(ltv);  %initial log asset value of borrower at origination 
  sigf = sqrt(rho)*sig;
  HN = H*(N/T);                %maturity in N time; 
  szfs = size(fs,1); 
  fs = [fs; fs; fs];           %use second and third block for numerical derivative
  Nsim1 = size(fs,1);
  
  %Euler discretization 
  f = [zeros(Nsim2,1) cumsum( (r-d-0.5*sig^2)*(T/N) + sigf*sqrt(T/N)*w , 2)];   
  f1 = f(:,N+1);       %from start of first loan to maturity of first loan w/ shocks until t removed
  
  %add fs shocks after loan origination
  fsa(1,:) = fs*sigf*sqrt(T);
  dstep = 10;     %1/dstep of SD step to evaluate numerical derivative
  df = sigf/dstep ; 
  fsa = fsa + df*cat(2,zeros(Nsim2,szfs),ones(Nsim2,szfs),-ones(Nsim2,szfs));   %divide by 2 to make comparable with average fs shock with multiple cohorts
  f1j = repmat(f1,[1,Nsim1]) + fsa;  
  
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
 
  %payoffs at loan maturities 
  L1 = LoanPayoff(F, f1j, ival, rho, sig, T);
  face1 = ones(size(f1j))*F;
  
  %factor realizations at t and t+H 
  ft = fsa + ival; 
  fH1 = f(:,1+HN);    
  fH1j = repmat(fH1,[1,Nsim1]) + fsa + ival;  
  FH1j = exp(fH1j).* exp(0.5*(1-rho)*H*sig^2); 
  Ft = exp(ft);   
  
  %get conditional payoff distribution based on t+H information  
  FHr1 = FH1j;      
  Lr1 = L1; 
  
  LHj = zeros(Nsim2,Nsim1); 
  FHr = reshape(FHr1(:,:),Nsim2*Nsim1,1);  
  Lr = reshape(Lr1(:,:),Nsim2*Nsim1,1); 
  [sortF, ind] = sort(FHr);
  sortL = Lr(ind);
  win = (Nsim2*Nsim1)/20;    %/10 seems to give about sufficient smoothness 
  LHs = fftsmooth(sortL,win); 
  newInd = zeros(size(ind)); 
  newInd(ind) = 1:length(FHr);
  %LHsn = reshape(LHs(newInd),Nsim2,1,Nsim1).*sc(:,j,:) - (face1(:,j,:)+face2(:,j,:))*(exp(y*H) - 1);
  LH1j = reshape(LHs(newInd),Nsim2,Nsim1);
    
   
  %get portfolio payoff distribution at t+H
  LH = LH1j.*exp(-r*(T-H)) ; 
  FH = FHr1;  
  face = face1.*exp(-r*(T-H));  
  
  BH = min(D,LH*exp(-y*H)); 
  EHex = LH*exp(-y*H) - BH; 
  EH = LH-BH;
  GH = g*max(D-LH*exp(-y*H),0); 
  
 
  %now integrate conditional on f_t over factor distribution at t+H 
  Lt = mean(LH)'*exp(-r*H); 
  Bt = mean(BH)'*exp(-r*H); 
  Et = mean(EH)'*exp(-r*H); 
  Gt = mean(GH)'*exp(-r*H); 
  mFt = mean(Ft)'; 
 
  %RN default indicator 
  def = ones(size(EHex)); 
  ldef = EHex > 0; 
  def(ldef) = 0; 
  
  mdef = mean(def)'; 

  %take numerical first derivative to get instantaneous equity vol
  %consider SD/dstep move of f   
  sigEt = (dstep/2)*(log(Et(szfs+1:2*szfs,1)) - log(Et(2*szfs+1:3*szfs,1)));  %since move is 2 times SD/10
  sigLt = (dstep/2)*(log(Lt(szfs+1:2*szfs,1)) - log(Lt(2*szfs+1:3*szfs,1)));  %since move is 2 times SD/10
  
  Lt = Lt(1:szfs,1); 
  Bt = Bt(1:szfs,1); 
  Et = Et(1:szfs,1); 
  LH = LH(:,1:szfs); 
  BH = BH(:,1:szfs); 
  EH = EH(:,1:szfs); 
  FH = FH(:,1:szfs); 
  mFt = mFt(1:szfs,1); 
  def = def(:,1:szfs);
  mdef = mdef(1:szfs,1);
  face = face(:,1:szfs);
  Gt = Gt(1:szfs,1); 
  
  
  
