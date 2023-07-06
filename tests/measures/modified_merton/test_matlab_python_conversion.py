import unittest
import os
import pathlib
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")


class MatlabPythonConversionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        frds_path = [
            i for i in pathlib.Path(__file__).parents if i.as_posix().endswith("frds")
        ].pop()
        self.mp = frds_path.joinpath(
            "src", "frds", "measures", "modified_merton", "matlab"
        ).as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.mp, nargout=0)

    def make_matlab_code(self, func_name: str, func_string: str):
        with open(pathlib.Path(self.mp).joinpath(f"{func_name}.m"), "w+") as f:
            f.write(func_string)

    def tearDown(self) -> None:
        for root, dirs, files in os.walk(self.mp):
            for f in files:
                if f.startswith("FRDSTest"):
                    os.remove(os.path.join(root, f))

    @unittest.skip
    def test_random_normal(self):
        np.random.seed(1)
        d0 = 20
        d1 = 1000
        normrand_py = norm.ppf(np.random.rand(d1, d0).T, 0, 1)

        self.make_matlab_code(
            "FRDSTestRRNG",
            """function [w] = FRDSTestRRNG()
            rng(1,'twister');
            d0 = 20; d1 = 1000;
            w = norminv(rand(d0, d1),0,1);""",
        )

        normrand_mt = self.eng.FRDSTestRRNG()
        assert_array_almost_equal(normrand_py, normrand_mt, 9)

    @unittest.skip
    def test_arange(self):
        fs = np.arange(-0.8, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
        fs = fs.reshape(-1, 1)

        self.make_matlab_code(
            "FRDSTest",
            """function [fs] = FRDSTest()
            fs = [-0.8:0.05:0.8]'/(0.2*sqrt(0.5)*sqrt(10)); 
            """,
        )

        mat = self.eng.FRDSTest()
        mat = np.asarray(mat)
        assert_array_almost_equal(fs, mat, 9)

    @unittest.skip
    def test_repmat(self):
        N = 10
        Nsim1 = 99
        Nsim2 = 1000
        rmat = np.arange(N)
        rmat = np.tile(rmat.reshape(N, 1, order="F"), (Nsim2, 1, Nsim1))
        self.make_matlab_code(
            "FRDSTest",
            """function [out] = FRDSTest(N, Nsim1, Nsim2)
            rmat(1,:,1) = [0:1:N-1];
            out = repmat(rmat, [Nsim2,1, Nsim1]);
            """,
        )
        mat = self.eng.FRDSTest(N, Nsim1, Nsim2)
        mat = np.asarray(mat)
        assert_array_almost_equal(rmat, mat, 9)

    @unittest.skip
    def test_data_generation(self):
        N = 10
        Nsim1 = 99
        Nsim2 = 1000
        T = 10
        H = 5
        HN = int(H * (N / T))

        aHmat = np.concatenate(
            (np.arange(HN, -1, -1) / N, np.arange(N - 1, HN, -1) / N)
        )
        aHmat = np.tile(aHmat.reshape(N, 1, order="F"), (Nsim2, 1, Nsim1))

        self.make_matlab_code(
            "FRDSTest",
            """function [aHmat] = FRDSTest(N, Nsim1, Nsim2, H, T)
            HN = H*(N/T);
            aHmat(1,:,1) = [HN:-1:0 N-1:-1:HN+1]/N;  
            aHmat = repmat(aHmat, [Nsim2,1, Nsim1]);  
            """,
        )
        # NOTE: N, H, T must be double, not int
        mat = self.eng.FRDSTest(
            matlab.double(N), Nsim1, Nsim2, matlab.double(H), matlab.double(T)
        )
        mat = np.asarray(mat)
        assert_array_almost_equal(aHmat, mat, 9)

    def test_reshape(self):
        N = 10
        Nsim1 = 99
        Nsim2 = 1000
        np.random.seed(1)
        ar = norm.ppf(np.random.rand(Nsim2, Nsim1).T, 0, 1)
        ar = np.reshape(ar, (Nsim2, Nsim1), order="F")

        self.make_matlab_code(
            "FRDSTest",
            """function [out] = FRDSTest()
            Nsim1 = 99; Nsim2 = 1000;
            rng(1,'twister');
            w = norminv(rand(Nsim1, Nsim2),0,1);
            out = reshape(w,Nsim2,1,Nsim1);
            """,
        )

        mat = self.eng.FRDSTest()
        mat = np.asarray(mat).squeeze()
        assert_array_almost_equal(ar, mat, 9)

    @unittest.skip
    def test_conversion_1(self):
        # fmt: off
        fs = np.arange(-0.8, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
        fs = fs.reshape(-1, 1)

        N = 10        # number of loan cohorts
        Nsim2 = 1000  # number of simulated factor realization paths (10,000 works well)

        r = 0.01      # risk-free rate
        d = 0.005     # depreciation rate of borrower assets
        y = 0.002     # bank payout rate as a percent of face value of loan portfolio
        T = 10        # loan maturity
        bookD = 0.63
        H = 5         # bank debt maturity
        D = bookD * np.exp(r * H) * np.mean(np.exp(r * np.arange(1, T+1)))   # face value of bank debt: as if bookD issued at start of each cohort, so it grows just like RN asset value
        rho = 0.5
        sig = 0.2
        ltv = 0.66  # initial ltv
        g = 0.5     # prob of govt bailout (only used for bailout valuation analysis)

        j = 3
        bookF = 0.6+j*0.02 # initial loan amount = book value of assets (w/ coupon-bearing loans the initial amount would be book value) 


        param = [r, T, bookF,  H, D, rho, ltv, sig, d, y, g]

        def func(fs, param, N, Nsim2):
            r = param[0]  # log risk-free rate
            T = param[1]  # original maturity of bank loans
            bookF = param[2]  # cash amount of loan issued = book value for a coupon-bearing loan issued at par
            H = param[3]  # bank debt maturity
            D = param[4]  # face value of bank debt
            rho = param[5]  # borrower asset value correlation
            ltv = param[6]  # initial LTV
            sig = param[7]  # borrower asset value volatility
            d = param[8]  # depreciation rate of borrower assets
            y = param[9]  # bank payout rate

            if len(param) > 10:
                g = param[10]
            else:
                g = 0

            rng = np.random.RandomState(1)
            w = norm.ppf(rng.rand(3*N, Nsim2).T, 0, 1)

            ival = np.log(bookF) - np.log(ltv)
            sigf = np.sqrt(rho) * sig
            HN = H * (N / T)  # maturity in N time
            if isinstance(HN, float):
                assert HN.is_integer()
                HN = int(HN)
            szfs = fs.shape[0]
            fs = np.concatenate((fs, fs, fs), axis=0)
            Nsim1 = fs.shape[0]

            rmat = np.arange(N)
            rmat = np.tile(rmat.reshape(N, 1, order="F"), (Nsim2, 1, Nsim1))
            ind1 = rmat >= HN
            ind2 = rmat < HN

            aHmat = np.concatenate((np.arange(HN, -1, -1) / N, np.arange(N - 1, HN, -1) / N))
            aHmat = np.tile(aHmat.reshape(N, 1, order="F"), (Nsim2, 1, Nsim1))

            atmat = (N - np.arange(N)) / N
            atmat = np.tile(atmat.reshape(N, 1, order="F"), (Nsim2, 1, Nsim1))

            f = np.concatenate(
                (
                    np.zeros((Nsim2, 1)),
                    np.cumsum((r - d - 0.5 * sig**2) * (T / N) + sigf * np.sqrt(T / N) * w, axis=1),
                ),
                axis=1,
            )

            fw = np.concatenate(
                (
                    np.zeros((Nsim2, 1)),
                    np.cumsum(-0.5 * rho * sig**2 * (T / N) + sigf * np.sqrt(T / N) * w, axis=1),
                ),
                axis=1,
            )

            f0w = np.tile(fw[:, N].reshape(fw[:, N].shape[0], 1, order='F'), (1, N)) - fw[:, 0:N]
            f1 = f[:, N : 2 * N] - f0w - f[:, 0:N]
            f2 = f[:, 2 * N : 3 * N] - f[:, N : 2 * N]

            fsa = (fs[np.newaxis, np.newaxis, :] * sigf * np.sqrt(T)).reshape((1, 1, Nsim1), order="F")
            dstep = 10  # 1/dstep of SD step to evaluate numerical derivative
            df = sigf / dstep
            fsa = np.repeat(fsa, N, axis=1) * atmat + df * np.concatenate(
                (
                    np.zeros((Nsim2, N, szfs)),
                    np.ones((Nsim2, N, szfs)),
                    -np.ones((Nsim2, N, szfs)),
                ),
                axis=2,
            )
            f1j = np.dstack((f1,) * Nsim1) + fsa
            f2j = np.dstack((f2,) * Nsim1)
            return f1j, f2j

        f1j, f2j = func(fs, param, N, Nsim2)

        self.make_matlab_code(
            "FRDSTest",
            """function [f1j, f2j] = FRDSTest(fs, param, N, Nsim2, w) 

            
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
            """
        )
        mat = self.eng.FRDSTest(
            matlab.double(fs), matlab.double(param), matlab.double(N), Nsim2,
            nargout=2
        )
        f1jmat = np.asarray(mat[0])
        f2jmat = np.asarray(mat[1])
        assert_array_almost_equal(f1j, f1jmat, 9)
        assert_array_almost_equal(f2j, f2jmat, 9)


if __name__ == "__main__":
    unittest.main()
