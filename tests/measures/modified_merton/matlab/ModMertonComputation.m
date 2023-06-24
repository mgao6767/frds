
function [FHr2, Lt, Bt, Et, LH, BH, EH, sigEt, mFt, def, mdef, face, FH, Gt, mu, F, sigLt] = ModMertonComputation(fs, param, N, Nsim2, w) 

  
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
 
  %payoffs at loan maturities 
  %T is the correct "maturity" below because idiosyncratic risk accumulated
  %since issuance of loan 
  %loan portfolio payoff at first maturity date (#sim  x cohort x fs shocks)
  %newmu coming out here is not exactly the same as mu because f1j doesn't
  %have shocks until time t
  L1 = LoanPayoff(F, f1j, ival, rho, sig, T);
  face1 = ones(size(f1j))*F;
    
  %for second generation of loans use ival2 which accounts for collateral
  %adjustment at loan roll over: same LTV as first generation loans, hence
  %same risk, same promised yield 
  %here newmu comes out identical to mu initially calculated above
  ival2 = log(L1/ltv);  
  F2 = L1*exp(mu*T); 
  L2 = LoanPayoff(F2, f2j, ival2, rho, sig, T);
  face2 = F2;
   
  face2(ind1) = 0; 
  face1(ind2) = 0; 
  
  %factor realizations at t and t+H
  %first by loan cohort, then average across cohorts 
  %relevant number is the asset value of loan portfolio in whole portfolio,
  %hence the Jensen's inequality term for the idio variance part 

  ft = repmat( repmat(f(:,N+1),1,N) - f0w - f(:,1:N), [1,1,Nsim1]) + fsa + ival;   %from start of first loan to time t  CHECK: no need for ival here because it's constant, makes no diff in conditioning below
                                                         %including shock fs since origination of loan
  fH1 = repmat(f(:,N+1+HN),1,N) - f0w - f(:,1:N);    %from start of first loan to to time t+H 
  fH2 = repmat(f(:,N+1+HN),1,N) - f(:,N+1:2*N);      %from start of second loan to time t+H 
  fH1j = repmat(fH1,[1,1,Nsim1]) + fsa + ival;   
  fH2j = repmat(fH2,[1,1,Nsim1]) + ival2;      %fs shock here goes into ival2  
  
  FH1j = exp(fH1j).* exp(0.5*(1-rho)*(T*aHmat)*sig^2); %idio part is uncertain over whole maturity (past and future) when conditioning on common factor 
  FH2j = exp(fH2j).* exp(0.5*(1-rho)*(T*aHmat)*sig^2); %but only accumulated dispersion until valuation point matters for Jensen's adj. 
 
  Ft = squeeze(mean(exp(ft).* exp(0.5*(1-rho)*(T*atmat)*sig^2),2));   %average across cohorts
  
  %get conditional payoff distribution based on t+H information  
  %i.e., conditional on factor path since inital orgination 
  %including any collateral amounts added or subtracted in the course of
  %roll over into the second generation of loans
  %use a (fast) smoother to compute this conditional expectation through
  %local averaging 
  %Conditional exp of loan payoff for rolled over loans is a function of
  %both 
  % (a) log factor shocks since roll over plus log collateral replenishment 
  % (b) the face value of the rolled over loan (which depends on factor
  %     realizations up to rollover date)
  %However, (b) scales both loan value and face value, so by descaling we
  %can reduce conditional expectation to one that depends only on (a) 
  sc = L1/bookF; 
  sc(ind1) = 1; 
  FHr1 = FH1j;      
  FHr1(ind2) = 0; 
  FHr2 = FH2j./sc;       
  FHr2(ind1) = 0; 
  Lr1 = L1; 
  Lr1(ind2) = 0; 
  Lr2 = L2./sc;  
  Lr2(ind1) = 0; 
  
  LHj = zeros(Nsim2,N,Nsim1); 
  for j = 1:N
    FHr = reshape(FHr1(:,j,:)+FHr2(:,j,:),Nsim2*Nsim1,1);  
    Lr = reshape(Lr1(:,j,:)+Lr2(:,j,:),Nsim2*Nsim1,1); 
    [sortF, ind] = sort(FHr);
    sortL = Lr(ind);
    win = (Nsim2*Nsim1)/20;    %/10 seems to give about sufficient smoothness 
    LHs = fftsmooth(sortL,win); 
    newInd = zeros(size(ind)); 
    newInd(ind) = 1:length(FHr);
    LHsn = reshape(LHs(newInd),Nsim2,1,Nsim1).*sc(:,j,:);
    LHj(:,j,:) = LHsn; 
    
  end   
  
   
  %integrate over cohorts and discount to get portfolio payoff distribution at t+H
  LH1j = LHj; 
  LH1j(ind2) = 0;  
  LH2j = LHj; 
  LH2j(ind1) = 0; 
  
  LH = squeeze(mean( LH1j.*exp(-r*(rmat-HN)*(T/N)) + LH2j.*exp(-r*(rmat-HN+N)*(T/N)) , 2)); 
  FH = squeeze(mean( FHr1 + FHr2 , 2 ));  
  
  face = squeeze(mean(face1.*exp(-r*(rmat-HN)*(T/N)) + face2.*exp(-r*(rmat-HN+N)*(T/N)), 2));  
  
  BH = min(D,LH*exp(-y*H)); 
  EHex = LH*exp(-y*H) - BH; 
  EH = LH-BH;
  GH = g*max(D-LH*exp(-y*H),0); 
  
 
  %now integrate conditional on f_t over factor distribution at t+H 
  %simply taking mean here works because no factor shocks until t 
  %so factor paths are all the same until t 
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
  %this is ok even though future replenishment/removal of f as collateral 
  %here just SD/dstep move in only source of stochastic shocks
  %of interest here is how this feeds through to Et (including through dampening collateral replenishment/removal)
  %(if one looked at an SD move in the dampened asset value, it would have a correspondingly  
  % bigger derivative) 
  
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
  
  
  
