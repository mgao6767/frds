


clear all;
close all; 
clc;

%output directory 
direc = ['./']; 

comp = 1;  %1 to re-calculate, 0 to use already saved file;  

fs = [-0.8:0.05:0.8]'/(0.2*sqrt(0.5)*sqrt(10)); %range for dW_0 shocks

N = 10;               %number of loan cohorts   
Nsim2 = 10000;        %number of simulated factor realization paths (10,000 works well) 

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
   
      
for j = 3; %1:10  % j = 3 for the plots in the main part of the paper
   
   j 
   bookF = 0.6+j*0.02; %initial loan amount = book value of assets (w/ coupon-bearing loans the initial amount would be book value) 
   rep = int2str(j); 
   
   if comp == 1 
    
  
      param = [r; T; bookF; H; D; rho; ltv; sig; d; y; g]; 
   
      [Lt, Bt, Et, LH, BH, EH, sigEt, mFt, def, mdef, face, FH, Gt, mu, F, sigLt] = ModMertonComputation(fs, param, N, Nsim2); 

      save([direc,'Simoutput'], 'Lt', 'Bt', 'Et', 'LH', 'BH', 'EH', 'sigEt', ...
        'mFt', 'def', 'mdef', 'face', 'FH', 'Gt', 'mu', 'F',...
           'Nsim2', 'N', 'param');
       
      
      
   else 
    
      load([direc,'Simoutput'])
    
      r = param(1); 
      T = param(2); 
      F = param(3); 
      H = param(4); 
      D = param(5); 
      rho = param(6); 
      ltv = param(7); 
      sig = param(8); 
      d = param(9); 
      y = param(10); 
      g = param(11); 
    
   end 

   [M,j0] = min(abs(mFt-exp(r*T/2))); 
   %j0 = find(abs(mFt-1) < 0.01);
   mface = max(face(:,j0))*exp(-r*((T-H)/2));  %approx. 
 
   
   options = optimset('TolFun',1.0e-16,'MaxIter',100000, 'TolX',1.0e-16, ...
               'MaxFunEvals',100000, 'Display', 'iter');

   K = size(sigEt,1); 
   mertdd = zeros(K,1); 
   mertdef = zeros(K,1); 
   mertGG = zeros(K,1); 
   mertsp = zeros(K,1); 
   mertA = zeros(K,1); 
   merts = zeros(K,1); 
   xmertdd = zeros(K,1); 
   xmertdef = zeros(K,1);
   mdefsingle1 = zeros(K,1);
   mFtsingle1 = zeros(K,1);
   Ltsingle1 = zeros(K,1);
   Ltsingle2 = zeros(K,1);
   bookFsingle2 = zeros(K,1);
   mdefsingle2 = zeros(K,1);
   mFtsingle2 = zeros(K,1);
   bookFsingle3 = zeros(K,1);
   mdefsingle3 = zeros(K,1);
   mFtsingle3 = zeros(K,1);
   Ltsingle3 = zeros(K,1);
   
   Et1 = []; 
   sigEt1 = []; 
   fs1 = []; 
   bookF1 = []; 
   
   
   %
   % Merton model fit to equity value and volatility from our model 
   % 
   
   for k = 1:K
       
      initb = [Et(k,1)+D*exp(-r*H); sigEt(k,1)/2];
      [bout] = fsolve(@(b) MertonSolution(b, Et(k,1), D, r, y, H, sigEt(k,1)), initb, options); 
      A = bout(1); 
      s = max(bout(2),0.0001); 
      mertdd(k,1) = (log(A)-log(D)+(r-y-s^2/2)*H)/(s*sqrt(H));   % RN distance to default over horizon H; 
                                                              % no subtraction of d because these are bank, not borrower assets
      mertdef(k,1) = normcdf(-mertdd(k,1)); 
      [C, P] = blsprice(A , D, r, H, s, y);   %payout rate y here just approx
      mertGG(k,1) = g*P; 
      mertsp(k,1) = (1/H)*log((D*exp(-r*H))/((D*exp(-r*H))-P)); 
      mertA(k,1) = A; 
      merts(k,1) = s; 
      
   end 
   
   mktCR = Et./(Et+Bt);
   sp = (1/H)*log((D*exp(-r*H))./Bt);     
      
   %
   % save files for different bookF values for MonotonicityAnalysis.m 
   %

   save([direc,'SimRNPD',rep], 'mertdef', 'mdef', 'bookF', 'sp', 'mertsp', 'fs', 'mFt',...
          'Nsim2', 'N', 'param');

   
   %
   % fit alternative (simplified) models
   %
  
   if j == 3    %main case discussed in the paper for one specific bookF
  
      %for alternative model (1):  
      %since both the model that produced Et and sigEt and the model that
      %we are using to try to fit these Et and sigEt are based on numerical
      %approximation, gradient-based methods doesn't work for fitting
      %Therefore, work with interpolant to find fs, bookF combinations that fit 
      %equity value and volatility 
        
      
      % (1) Perfectly correlated borrowers with overlapping cohorts
      
      rho = 0.99;  %approx to rho = 1, code can't handle rho = 1;  
      sig = 0.20*sqrt(0.5); 
      sfs = [-2.6:0.05:0.8]'/(0.2*sqrt(0.5)*sqrt(10)); %[-0.8:0.05:0.8]
 
      %we have D = bookD*exp(r*H)*mean(exp(r*([1:T])')) here
      %and D = bookD*exp(xr(1,j,k,q)*H); in ModMertonCreateLookup, 
      %so adjust bookD1 here
      bookD1 = bookD*mean(exp(r*([1:T])'));      
      [xfs,xsig,xr,xF] = ndgrid( sfs, sig, r, 0.4:0.01:1.44);
       
      [xLt, xBt, xEt, xFt, xmdef, xsigEt] = ModMertonCreateLookup(d,y,T,H,bookD1,rho,ltv,xfs,xr,xF,xsig,N,Nsim2); 
      xxsigEt = squeeze(xsigEt); 
      xxEt  = squeeze(xEt);
      xxmdef = squeeze(xmdef); 
      xxFt = squeeze(xFt); 
      ymdef = scatteredInterpolant(xxEt(:),xxsigEt(:),xxmdef(:),'natural','none');
      yFt = scatteredInterpolant(xxEt(:),xxsigEt(:),xxFt(:),'natural','none');
      mdefsingle1 = ymdef(Et,sigEt);
      mFtsingle1 = yFt(Et,sigEt); 
      %recomputing Et, sigEt based on fs1, bookF1 gets back Et, sigEt to
      %a very good approximation 
      
      
      % (2) Single cohort of borrowers
      
      T = 5; 
      rho = 0.5; sig = 0.2;  
      [xfs,xsig,xr,xF] = ndgrid( sfs, sig, r, 0.4:0.01:1.44);
      [xLt, xBt, xEt, xFt, xmdef, xsigEt] = ModSingleMertonCreateLookup(d,y,T,H,bookD1,rho,ltv,xfs,xr,xF,xsig,N,Nsim2); 
      xxsigEt = squeeze(xsigEt); 
      xxEt  = squeeze(xEt);
      xxmdef = squeeze(xmdef); 
      xxFt = squeeze(xFt);
      ymdef = scatteredInterpolant(xxEt(:),xxsigEt(:),xxmdef(:),'natural','none');
      yFt = scatteredInterpolant(xxEt(:),xxsigEt(:),xxFt(:),'natural','none');
      mdefsingle2 = ymdef(Et,sigEt);
      mFtsingle2 = yFt(Et,sigEt); 
     
      
      % (3) Single (or perfectly correlated) borrower model 
      
      T = 5; 
      rho = 0.99; sig = 0.2*sqrt(0.5);   %approx of perfect correlation
      [xfs,xsig,xr,xF] = ndgrid( sfs, sig, r, 0.4:0.01:1.44);
      [xLt, xBt, xEt, xFt, xmdef, xsigEt] = ModSingleMertonCreateLookup(d,y,T,H,bookD1,rho,ltv,xfs,xr,xF,xsig,N,Nsim2); 
      xxsigEt = squeeze(xsigEt); 
      xxEt  = squeeze(xEt);
      xxmdef = squeeze(xmdef); 
      xxFt = squeeze(xFt); 
      ymdef = scatteredInterpolant(xxEt(:),xxsigEt(:),xxmdef(:),'natural','none');
      yFt = scatteredInterpolant(xxEt(:),xxsigEt(:),xxFt(:),'natural','none');
      mdefsingle3 = ymdef(Et,sigEt);
      mFtsingle3 = yFt(Et,sigEt); 
     
      
      % (4) Merton model with asset value and asset volatility implied our modified model 
     
      for k = 1:K
         A = Lt(k,1); 
         s = sigLt(k,1); 
         xmertdd(k,1) = (log(A)-log(D)+(r-y-s^2/2)*H)/(s*sqrt(H));                                                       
         xmertdef(k,1) = normcdf(-xmertdd(k,1)); 
      end
   
     
      %
      % Some comparisons and figures for main part of the paper
      % 
 
      jup = j0+8;
      jdn = j0-8; 
      disp('shock') 
      disp([fs(j0) fs(jup)])
      disp('Agg Borrower Asset Value') 
      disp([mFt(j0) mFt(jup)])
      disp('Market equity ratio') 
      disp([mktCR(j0) mktCR(jup)])
      disp('Mod RN def prob') 
      disp([mdef(j0) mdef(jup)])
      disp('Merton RN def prob') 
      disp([mertdef(j0) mertdef(jup)])
      disp('Mod Credit spread') 
      disp([sp(j0) sp(jup)])
      disp('Merton Credit spread') 
      disp([mertsp(j0) mertsp(jup)])


      A = [mFt(j0) mFt(jup) mFt(jdn); ...
        Lt(j0) Lt(jup) Lt(jdn); ...
        mktCR(j0) mktCR(jup) mktCR(jdn); ...
        mdef(j0) mdef(jup) mdef(jdn); ...
        sp(j0)*100 sp(jup)*100 sp(jdn)*100; ...
        -9999 -9999 -9999;  ...
        mertdef(j0) mertdef(jup) mertdef(jdn); ...
        mertsp(j0)*100 mertsp(jup)*100 mertsp(jdn)*100];  
      Af = [' ' ' ' ' '];
      Aform = [Af; Af; Af; Af; Af; Af; Af; Af; Af]; 
      row = {'Agg. Borrower Asset Value', 'Bank Asset Value', 'Bank Market Equity/Market Assets', 'Bank 5Y RN Default Prob.', 'Bank Credit Spread (\%)', ' ', 'Merton 5Y RN Default Prob.', 'Merton Credit Spread (\%)'};
      head = {'Borrower asset value const.', 'Borrower asset value $+30\%$', 'Borrower asset value $-30\%$'};
      savename = [direc,'SimSummary.tex'];
      matrix2latex_Stefan(A, savename, 'rowLabels', row, 'columnLabels',...
            head, 'alignment', 'c', 'format', '%6.2f', 'size', 'footnotesize', 'specform', Aform);         
       

      f = figure
      subplot(3,1,1)
      hold on
      scatter(FH(:,j0), LH(:,j0)+face(:,j0)*(exp(y*H) - 1),5)  %LH is net of dividends, add back for this plot 
      line([0,mface],[0,mface],'LineWidth',2,'Color','r', 'LineStyle', '--')
      line([mface,2.5],[mface,mface],'LineWidth',2,'Color','r','LineStyle', '--')
      set(gca,'XLim',[0 2.5])
      set(gca,'YLim',[0 1.2])
      title('(a) Bank asset value') 
      ylabel('Bank asset value','FontSize',12) 
      %xlabel('Aggregate borrower asset value','FontSize',12) 
      hold off  
      subplot(3,1,2)
      hold on
      scatter(FH(:,j0),EH(:,j0),6)
      line([0,D],[0,0],'LineWidth',2,'Color','r', 'LineStyle', '--')
      line([D,mface],[0,mface-D],'LineWidth',2,'Color','r', 'LineStyle', '--')
      line([mface,2.5],[mface-D,mface-D],'LineWidth',2,'Color','r', 'LineStyle', '--')
      set(gca,'XLim',[0 2.5])
      %set(gca,'YLim',[0 1])
      title('(b) Bank equity value') 
      ylabel('Bank equity value','FontSize',12) 
      %xlabel('Aggregate borrower asset value','FontSize',12) 
      hold off
      subplot(3,1,3) 
      hold on
      scatter(FH(:,j0),BH(:,j0),6)
      line([0,D],[0,D],'LineWidth',2,'Color','r', 'LineStyle', '--')
      line([D,2.5],[D,D],'LineWidth',2,'Color','r', 'LineStyle', '--')
      %set(gca,'XLim',[0 2])
      set(gca,'XLim',[0 2.5])
      title('(c) Bank debt value') 
      ylabel('Bank debt value','FontSize',12) 
      xlabel('Aggregate borrower asset value','FontSize',12) 
      hold off
      set(gcf, 'PaperSize', [6 7]);
      set(gcf, 'PaperPositionMode', 'manual');
      %set(gcf, 'PaperPosition', [0 0 5.5 8]);
      set(gcf, 'PaperUnits', 'normalized')
      set(gcf, 'PaperPosition', [0 0 1 1])
      eval(['print(''',direc,'PayoffsAtDMat'',''-dpdf'')'])


     f= figure
     scatter(mFt,mktCR)

     f= figure
     scatter(mFt,Et)
     ylabel('Bank equity value','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'mVe.pdf'')']);

     f= figure
     hold on
     scatter(mFt,mdef,'MarkerEdgeColor','b')
     scatter(mFt,mertdef,'MarkerEdgeColor','r', 'Marker', '+')
     hold off
     legend('Actual','Merton Model') 
     ylabel('RN bank default probability','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'mdef.pdf'')']);

     f = figure
     hold on
     scatter(mFt,mdef,'MarkerEdgeColor','b')
     scatter(mFt,mertdef,'MarkerEdgeColor','r', 'Marker', '+')
     hold off
     legend('Actual','Merton Model') 
     set(gca,'YLim',[0 0.5])
     set(gca,'XLim',[0.6 1.6])
     ylabel('RN bank default probability','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'mdef2.pdf'')']);

     
     f = figure 
     hold on
     scatter(mFt,mdef,'MarkerEdgeColor','b')
     plot(mFt,mdefsingle2,'--k')
     plot(mFt,mdefsingle3,'--r')
     scatter(mFt,mertdef,'MarkerEdgeColor','r', 'Marker', '+')
     hold off
     lgd = legend('Actual','Single cohort','Single borrower','Merton Model')
     lgd.FontSize = 12;
     ylabel('RN bank default probability','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'mdefsingles.pdf'')']);
   
      
     
     f = figure 
     hold on
     scatter(mFt,mdef,'MarkerEdgeColor','b') 
     plot(mFt,xmertdef,'--k')
     scatter(mFt,mertdef,'MarkerEdgeColor','r', 'Marker', '+') 
     hold off
     lgd = legend('Actual','Merton w/ actual asset value and volatility','Merton Model')
     lgd.FontSize = 12;
     ylabel('RN bank default probability','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'mertalt.pdf'')']);
   
      
     f = figure
     hold on
     scatter(mFt,Gt,'MarkerEdgeColor','b')
     scatter(mFt,mertGG,'MarkerEdgeColor','r', 'Marker', '+')
     hold off
     legend('Actual','Merton Model') 
     set(gca,'YLim',[0 0.05])
     set(gca,'XLim',[0.6 1.6])
     ylabel('Value of government guarantee','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'mgg.pdf'')']);

     f = figure
     hold on
     scatter(mFt,sp,'MarkerEdgeColor','b')
     scatter(mFt,mertsp,'MarkerEdgeColor','r', 'Marker', '+')
     hold off
     legend('Actual','Merton Model') 
     set(gca,'YLim',[0 0.03])
     set(gca,'XLim',[0.6 1.6])
     ylabel('Credit spread','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'msp.pdf'')']);

     f = figure
     scatter(mFt,sigEt)
     ylim([0 0.7]) 
     ylabel('Inst. bank equity vol.','FontSize',12) 
     xlabel('Aggregate borrower asset value','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'sigVe.pdf'')']);

     %construct Merton model equity volatility with constant asset volatility
     [Cdelta, Pdelta] = blsdelta(Lt , D, r, H, 0.03, y);
     [C, P] = blsprice(Lt , D, r, H, 0.03, y); 
     xeqvol = 0.03*Cdelta.*(Lt./C); 
     xlev = Lt./C; 


     f = figure
     hold on
     %scatter(log(Lt./Et),log(sigEt))
     %scatter(log(Lt./Et),log((Lt./Et)*merts(17)),'MarkerEdgeColor','r', 'Marker', '+') 
     scatter(log(Lt./Et),sigEt)
     scatter(log(xlev(4:end)),xeqvol(4:end),'MarkerEdgeColor','r', 'Marker', '+') %=if asset vol was the same as in our model
     hold off
     xlim([0 4]) 
     ylabel('Inst. bank equity vol.','FontSize',12) 
     xlabel('Bank assets/equity','FontSize',12) 
     eval(['saveTightFigure(f,''',direc,'sigE.pdf'')']);

     figure
     scatter(sigEt,Et)

     f = figure
     hold on
     line([0,0.8],[0,0.8],'LineWidth',2,'Color','r')
     line([0.8,2],[0.8,0.8],'LineWidth',2,'Color','r')
     line([0,0.6],[0,0],'LineWidth',2,'Color','b', 'LineStyle', ':')
     line([0.6,0.8],[0,0.2],'LineWidth',2,'Color','b', 'LineStyle', ':')
     line([0.8,2],[0.2,0.2],'LineWidth',2,'Color','b', 'LineStyle', ':')
     line([0,0.6],[0,0.6],'LineWidth',2,'Color','g', 'LineStyle', '--')
     line([0.6,2],[0.6,0.6],'LineWidth',2,'Color','g', 'LineStyle', '--')
     text(1.2, 0.25  , 'Bank equity', 'Color', 'k','FontSize',16);
     text(1.2, 0.65  , 'Bank debt', 'Color', 'k','FontSize',16);
     text(1.2, 0.85  , 'Bank assets', 'Color', 'k','FontSize',16);
     ylabel('Payoff','FontSize',16) 
     xlabel('Borrower asset value','FontSize',16) 
     set(gca,'FontSize',16)
     set(gca,'XLim',[0 2])
     set(gca,'YLim',[0 1])
     hold off  
     eval(['saveTightFigure(f,''',direc,'payoffs.pdf'')']);

     x = 0:0.01:2; 
     f = figure
     hold on
     plot(x,1.6*(normcdf(x*2)-0.5),'LineWidth',2,'Color','r') 
     hold off 
     text(0.45, 0.4  , 'Bank asset value', 'Color', 'r','FontSize',16);
     ylabel('Bank asset value','FontSize',16) 
     xlabel('Borrower asset value','FontSize',16) 
     set(gca,'FontSize',16)
     set(gca,'XLim',[0 2])
     set(gca,'YLim',[0 1])
     hold off  
     eval(['saveTightFigure(f,''',direc,'payoffs1.pdf'')']);

     dy = [1.6*(normcdf(1*2)-0.5) - 2*(normpdf(1*2))*0.7; 1.6*(normcdf(1*2)-0.5) + 2*(normpdf(1*2))*0.7];
     dx = [0.3; 1.7]; 
     x = 0:0.01:2; 
     f = figure
     hold on
     plot(x,1.6*(normcdf(x*2)-0.5),'LineWidth',2,'Color','r') 
     line(dx,dy,'LineWidth',2,'Color','k', 'LineStyle', ':')
     hold off 
     text(0.45, 0.4  , 'True nonlinear bank asset value', 'Color', 'r','FontSize',16);
     text(0.1, 0.85  , 'Locally fitted lognormal model', 'Color', 'k','FontSize',16);
     ylabel('Bank asset value','FontSize',16) 
     xlabel('Borrower asset value','FontSize',16) 
     set(gca,'FontSize',16)
     set(gca,'XLim',[0 2])
     set(gca,'YLim',[0 1])
     hold off  
     eval(['saveTightFigure(f,''',direc,'payoffs2.pdf'')']);

   end 

end






