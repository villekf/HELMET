%% This script is the "main" script for the KF toolbox. All necessary and optional parameters are selected in this sript.
% Note: All manual input values, when they are NOT sparse, should be of
% single precision (otherwise you'll get erroneous results). Sparse inputs
% should be MATLAB/Octave sparse matrices. For more details on the various
% options see the documentation.

%%%%%%%%%%%%%% KF Type %%%%%%%%%%%%%%
% This specifies the type of KF algorithm used. Many of these algorithms
% have their own specific properties that can be adjusted below.
% 0 = Regular (discrete) Kalman filter (KF)
% 1 = Information filter
% 2 = One step KF
% 3 = Ensemble Kalman filter (EnKF)
% 4 = Alternate EnKF
% 5 = Cholesky EnKF
% 6 = Square root EnKF
% 7 = Ensemble transform Kalman filter
% 8 = Error subspace transform Kalman filter
% 9 = Dimension reduction KF
% 10 = Conjugate gradient KF
% 11 = Variational CG KF
% 12 = Spectral KF
options.algorithm = 0;

%%%%%%%%%%%%%% Complex data type %%%%%%%%%%%%%%
% Here you can specify whether the input measurement data is complex or not
% and how to compute the KF estimates when using complex data
% 0 = No complex data (all data is real valued)
% 1 = Measurements and estimates are complex, separate estimates are
% computed for real and imaginary parts (system matrix is real valued)
% 2 = Measurements, estimates and covariances are complex, but separate
% estimates and covariance matrices are calculated for real and imaginary
% parts (system matrix is real valued)
% 3 = Both real and imaginary parts are computed at the same time;
% covariances, estimates and measurements contain both the real and
% imaginary parts. System matrix is used as [real(A) -imag(A);imag(A)
% real(A)].
% 4 = Everything is complex valued (does not support sparse matrices)
options.complexType = 0;

%%%%%%%%%%%%%% Regularization type %%%%%%%%%%%%%%
% Here you can select the type of (spatial) regularization used
% 0 = No regularization
% 1 = Augmented KF (AKF)
% 2 = Spatial prior KF (SPKF)
% 3 = Denoising type 1 (e.g. TVKF)
% 4 = Denoising type 2
options.regularization = 0;

%%% Observation noise variance for the augmented part with AKF
% Regularization type 1 ONLY
options.RR = 1;

%%% Regularization matrix type
% Regularization type 1, 2 and 4, and prior 7 (see below) ONLY
% 0 = 1D first order (forward) difference matrix
% 1 = 2D first order (forward) difference matrix
% 2 = 2D second order (forward) difference matrix
% 3 = Sum of 2D first order and 2D second order (forward) difference
% matrices (use the weighting coefficient below (Lscale) to control the
% strength of the 2nd order matrix)
% 4 = 3D first order (forward) difference matrix
% 5 = Use custom regularization matrix (can be full or sparse), input as
% options.L
options.Ltype = 0;

% ONLY options.Ltype = 3
% Weight of the 2nd order difference matrix when compared to the 1st order
% (1 means they are summed without any scaling, 0 means only first order
% component)
options.Lscale = 0.1;

%%% AKF/SPKF type
% Regularization type 1 and 2, and prior 7 (see below) ONLY
% 0 = No weighting based on the reference image (matrix L is used without
% any weighting)
% 1 = Weights based on the reference image (kappa) as in AKF and SPKF paper
% 2 = Weights based on the TV reference image weights as in TV-KF
options.augType = 0;

%%% Prior type
% Regularization type 3 ONLY
% The prior (denoising type) used
% 1 = TV type 1
% 2 = TV type 2
% 3 = APLS
% 4 = SATV
% 7 = Regularization matrix L (AKF/SPKF parameters affect this L, i.e.
% matrix type) 
% 8 = TGV denoising (does not support weighting matrix)
options.prior = 1;
% Denoising iterations
% The number of iterations for the denoising step in regularization types 3
% and 4
options.nIterDenoise = 50;
% TV smoothing parameter
% This value allows the TV to be differentiable (a small positive constant
% in the square root)
options.TVsmoothing = 1e-6;
options.weights = [sqrt(2);1;sqrt(2);1;4+sqrt(2)*4;1;sqrt(2);1;sqrt(2)];
% SATV Phi value
options.SATVPhi = 1;
% Eta value for APLS
options.eta = 1;
% Regularization parameter 1 for TGV
options.lambda1 = 0.001;
% Regularization parameter 2 for TGV
options.lambda2 = 0.002;
% Relaxation parameter for TGV
options.relaxationParameter = 1.99;
% Proximal value for TGV
options.proximalValue = 0.01;

%%% Include covariance matrix as a weighting matrix
% Regularization types 3 and 4 ONLY
% If set to false, the a posteriori error covariance is NOT included as the
% weighting matrix in denoising steps
% NOTE: Does not apply to any of the ensemble filters (algorithms 3-8)
options.includeCovariance = true;

%%% Step size
% Regularization type 4 ONLY
options.gamma = 0.005;

%%% Use reference image
% Applies to regularization types 3 and 4 ONLY
% If true, uses the reference image (given below) as a weight
options.useAnatomical = false;
% Use separate reference images
% If complex data is used, setting this to true will use separate reference
% images for real and imaginary parts. Otherwise, only the real part is
% used
options.complexRef = false;

%%% Reference image
% Needed when using any weighting based on the reference image. Applies to
% all regularization types if weighting is selected. Input any variable
% here that has the same total number of elements as the final estimate.
options.referenceImage = [];

%%% Edge threshold parameter
% Priors 1-2 ONLY and regularization matrix L
% This is used whenever the reference image based weighting is used.
% Controls the edge thresholding
options.C = 0.1;

%%% Regularization parameter
% Real part
options.beta = 1;
% Imaginary part (only when complex data is used)
options.betaI = 1;

%%%%%%%%%%%%%% Use Matern prior %%%%%%%%%%%%%%
% If true, then replaces the process noise covariance with the Matern
% prior. This setting overwrites all values you give to the process noise
% covariance.
options.matern = false;
% Sigma value for Matern prior
options.maternSigma = 0.01;
% Lambda value for Matern prior
options.maternLambda = 0.09;
% Nu-value
% Accepted values are 5/2, 3/2, 1/2 and Inf
options.maternNu = 5/2;
% l-value for Matern prior
options.maternl = 2;
% Delta T value for Matern prior
options.maternDeltaT = 0.0385;
% The maximum number of pixels away where the Euclidian distance is
% computed 
options.maternDistance = 5;


%%%%%%%%%%%%%% Fading memory / covariance inflation %%%%%%%%%%%%%%
% If value other than 1 is given, then the fading memory KF is used when
% using algorithm 0. For ETKF and ESTKF, this is the covariance
% inflation parameter.
% Empty value/array functions identical to 1
options.fadingAlpha = 1;

%%%%%%%%%%%%%% Steady state KF %%%%%%%%%%%%%%
% If true, computes the steady state Kalman gain and error covariance
% matrices
% NOTE: Algorithms 0-2 ONLY
options.steadyKF = false;
% Steady state threshold
% The iterations for the steady state gain are aborted after the Euclidean
% norm of the previous diagonal elements minus the current diagonal
% elements of the a posteriori covariance is below this value
options.steadyThreshold = 0.001;

%%% Precompute the error covariance
% You can specify the number of preiterations for the error covariance
% computation. If value greater than 0 is used, then the Kalman gain and
% error covariances are computed options.covIter times before the actual KF
% estimates are computed.
% NOTE: Applies only to algorithms 0-2 and algorithm 9
% NOTE: Steady state takes priority over this
options.covIter = 0;
% Precompute the initial value
% If true, will also compute the KF estimates as the initial values for the
% actual filter. This applies to either steady state or precomputed
% covariance, whichever is selected
options.computeInitialValue = false;

%%%%%%%%%%%%%% Use Kalman smoother (KS) %%%%%%%%%%%%%%
% If true, then Kalman smoother is used
% NOTE: Only algorithms 0-2, 7, 8, 9, 10 and 11 support KS
options.useKS = false;
% Number of time steps included in each smoother step
% If this equals to the total number of time steps, then only one smoothing
% step is performed. Otherwise smoothing steps are performed every
% options.sSkip intervals with options.nLag number of measurements in each
% step. See the documentation for example.
options.nLag = 50;
% The number of time steps to skip before another smoothing step
options.sSkip = 30;
% Use steady sate KS
% If true, then the steady state KS is used. This means that only a single
% KS gain is computed at each smoothing step.
options.steadyKS = false;
% Approximate KS gain
% If true, the a priori error covariance is approximated as a diagonal
% matrix in the KS gain computations
options.approximateKS = false;

%%%%%%%%%%%%%% Consistency tests %%%%%%%%%%%%%%
% Compute the consistency tests (NIS and autocorrelation by default)
% All previous selections will be applied normally.
% Only algorithms 0-2 and 10 are supported
options.computeConsistency = false;
% Number of preiteration steps
% This is the number of time steps that the entire filter is ran before the
% consistency tests are computed, i.e. the number of pre-iteration steps.
% If options.covIter > 0, the covariances are precomputed before these
% steps
options.consistencyIter = 610;
% Number of time average steps
% Time average NIS and autocorrelation will be computed with a total of
% this many time steps
options.consistencyLength = 610;
% Step difference in autocorrelation
% The time-average autocorrelation requires the innovation vectors to be j
% steps apart. This value corresponds to the j variable
options.stepSize = 610;
% Compute the Bayesian p-test
% if true, computes the Bayesian p-test in addition to the NIS and
% autocorrelation tests
% NOTE: Only algorithms 0 - 2 are supported
options.computeBayesianP = true;

%%%%%%%%%%%%%% Ensemble size %%%%%%%%%%%%%%
% For ensemble filters only (algorithms 4-8)
options.ensembleSize = 1000;
% Use ensemble mean
% If true, uses the mean of the ensemble to compute the new a priori
% ensemble. Otherwise uses the full ensemble.
options.useEnsembleMean = false;

%%%%%%%%%%%%%% Reduced rank KF (algorithm 9) %%%%%%%%%%%%%%
% Use custom prior covariance?
options.useCustomCov = false;
% Input the custom prior covariance here if the above is true
options.Sigma = [];
% The dimensions of the initial prior covariance (when useCustomCov =
% false)
options.covDimX = 100;
options.covDimY = 100;
options.covDimZ = 1;
% Sigma value of the prior covariance
% Valid only if custom covariance is not used
options.reducedCovSigma = 1;
% Correlation length
% Valid only if custom covariance is not used
options.reducedCorLength = 3;
% The number of basis functions to use
options.reducedBasisN = 500;

%%%%%%%%%%%%%% CG KF/VKF %%%%%%%%%%%%%%
% Number of CG iterations
options.cgIter = 40;
% Stopping threshold for CG iterations
options.cgThreshold = 1e-4;
% Force orthogonalization in CG/Lanczos iterations
% Increases computation times, but increases accuracy
options.forceOrthogonalization = true;

%%%%%%%%%%%%%% Sliding window KF %%%%%%%%%%%%%%
% The number of sliding time steps used. E.g., if options.window = 3, each
% KF time step contains three time steps. In the subsequent steps, the last
% time step is removed and a new one is added.
options.window = 1;

%%%%%%%%%%%%%% Estimate size %%%%%%%%%%%%%%
% Number of columns/rows
options.Nx = 128;
% Number of rows/columns
options.Ny = 128;
% Number of slices
options.Nz = 1;

%%%%%%%%%%%%%% 3D or pseudo-3D %%%%%%%%%%%%%%
% Applies only if options.Nz > 1
% If true, assumes that the data is full 3D --> system matrix is assumed to
% contain all the necessary rows for x = H'*y in 3D
% If false, assumes a pseudo-3D case where each slice is assumed to be an
% independent 2D reconstruction with the same system matrix for each slice
% (common in MRI)
options.use3D = false;

%%%%%%%%%%%%%% Number of time steps %%%%%%%%%%%%%%
options.Nt = 50;

%%%%%%%%%%%%%% Input measurement data %%%%%%%%%%%%%%
% Input the measurement data here
options.m0 = mgeom.projdata;

%%% Number of measurements per time step
% The number of measurements at each time step. Has to be > 0 and must be
% input. This refers to the number of elements in options.m0 for each time
% step.
options.Nm = 1;

%%%%%%%%%%%%%% Input system matrix %%%%%%%%%%%%%%
% Input your system/observation matrix here. The matrix can be either
% sparse or dense, as well as either real-valued or complex-valued. Note
% that dense (full) matrices need to be in single precision while sparse
% matrices need to be in MATLAB/Octave sparse format. This must be input.
options.H = [];

%%% Number of unique system matrix cycles
% The system/observation matrix contains (Nm * matCycles) * (Nx * Ny * Nz)
% elements. This also must be input.
options.matCycles = 1;

%%% Store all system matrix elements in the device
% If true, all system matrix elements are stored in the selected device.
% Otherwise, only the current time step is stored and all steps are loaded
% on the fly. Set this to false if you want to converse memory (slows down
% computations).
options.storeData = true;

%%%%%%%%%%%%%% Initial value for the a priori error covariance %%%%%%%%%%%%%%
% Can be either scalar or vector, in both cases the matrix will be diagonal
% with the vector case corresponding to the variances.
options.P0 = 1e-2;

%%% Store the a posteriori error covariance
% If set to 0, no covariance matrix is stored. If set to 1, the diagonals
% of the error covariance are stored at each time step. If set to 2, the
% entire final error covariance is stored. This matrix is then output along
% with the estimates.
% NOTE: Supports only algorithms 0-2 and 9. 9 only supports value 2.
options.storeCovariance = 0;

%%%%%%%%%%%%%% Initial value for the estimate %%%%%%%%%%%%%%
if options.complexType == 0
    options.x0 = zeros(options.Nx * options.Ny * options.Nz, 1,'single');
else
    options.x0 = complex(zeros(options.Nx * options.Ny * options.Nz, 1,'single'));
end

%%%%%%%%%%%%%% Process noise covariance values %%%%%%%%%%%%%%
% Can be either scalar, vector, matrix or cell matrix. In the case of
% scalar or vector inputs, Q is assumed to be diagonal. In the case of
% matrix input, columns are considered as diagonals for each time step. For
% cell matrix, each cell represents a single time step. For a constant
% matrix Q, use only one cell. Matrix Q can be either sparse or dense (the
% latter has to be in single precision).
options.Q = 5e-3;
%%% Process noise covariance as inverse
% Set this to true if the input values/matrices of Q have already
% been inverted (applies to algorithms that use inverted covariance)
options.invertedQ = false;

%%%%%%%%%%%%%% Observation noise covariance values %%%%%%%%%%%%%%
% Can be either scalar, vector, matrix or cell matrix. In the case of
% scalar or vector inputs, R is assumed to be diagonal. In the case of
% matrix input, columns are considered as diagonals for each time step. For
% cell matrix, each cell represents a single time step. For a constant
% matrix Q, use only one cell. Matrix R can be either sparse or dense (the
% latter has to be in single precision).
options.R = 4e-2;
%%% Observation noise covariance as inverse
% Set this to true if the input values/matrices of R have already
% been inverted (applies to algorithms that use inverted covariance)
options.invertedR = false;

%%%%%%%%%%%%%% Input state transition matrix %%%%%%%%%%%%%%
% You can include your own state transition matrix here.
% NOTE: if you use SPKF or the kinematic model, any F matrix you input here
% will be OVERWRITTEN.
% F can be either full or sparse matrix. If you want a time-varying state
% transition matrix, the input should be a cell array where each cell
% element corresponds to a state transition matrix in a single time step.
% options.F = yourOwnMatrix;

%%% Use kinematic model
% If true, uses the first order kinematic model
% NOTE: This overwrites the above state transition matrix
options.useKinematicModel = false;
% Delta T
% "Time" value used by the kinematic model
options.deltaT = 0.0385;

%%%%%%%%%%%%%% Input (optional) state vector %%%%%%%%%%%%%%
% You can include additional vector to be included to the a priori
% estimate, defined as the vector u in the equation list. If the vector u
% is time dependent, input it as a 2D matrix, where each column corresponds
% to each time step. This vector is completely optional.
% options.u = [];

%%% Input state transition matrix for u
% If the above u vector is used, you can input a separate state transition
% matrix for it
% options.G = [];

%%%%%%%%%%%%%% Backend type %%%%%%%%%%%%%%
% This specifies how the computations are performed. CUDA capable GPUs
% should use CUDA, otherwise OpenCL. CPUs can use either OpenCL (if
% runtimes are available) or CPU
% 0 = OpenCL
% 1 = CUDA
% 2 = CPU
options.backend = 0;

%%% Device used
% You can use the following commands (uncommented) to determine the various
% device numbers:
% ArrayFire_OpenCL_device_info() % OpenCL
% ArrayFire_CUDA_device_info() % CUDA
% ArrayFire_CPU_device_info() % CPU
options.device = 0;

%%

if ~issparse(options.H) && isa(options.H,'double')
    options.H = single(options.H);
end
[xt,xs,P] = reconstructKF(options);