Code for Bank Risk Dynamics and Distance to Default
by Stefan Nagel and Amiyatosh Purnanandam 
6/28/2019

*** These code files are made available for non-commercial purposes only ***

-------------------

(1) Simulation: 

main program: ModMertonSimulation.m 
calls ModMertonComputation.m to calculate modified Merton model RNPD for a range of dW_0 shocks 


(2) Empirical application: 

main program: ModMertonEmpirical_unobsbookF.m 
takes files with bank-level data on equity values/book debt, equity return volatility to compute modified Merton model RNPD by calling ModMertonCreateLookup.m (which calls ModMertonComputation.m) 