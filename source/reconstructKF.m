function [xt,xs, varargout] = reconstructKF(options)
%RECONSTRUCTKF Performs the KF toolbox reconstructions
%   This function computes the necessary precomputation steps for the
%   various KF algorithms and selections. Finally, the actual KF estimates
%   are computed with a separate mex-file.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022 Ville-Veikko Wettenhovi
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(options.H)
    error('The system matrix H is empty (options.H). This must be input.')
end
if isempty(options.Nm) || options.Nm <= 0
    error('The number of measurements per time step is empty, zero or negative. This must be input and has to be positive.')
end
if isempty(options.matCycles) || options.matCycles <= 0
    error('The number of unique system matrix cycles per time step is empty, zero or negative. This must be input and has to be positive.')
end
if options.complexType == 0 && ~isreal(options.m0)
    warning('Real data type selected yet the measurements contain complex data. Using only the real part')
    options.m0 = real(options.m0);
end
if options.complexType > 0
    options.complexMeasurements = true;
else
    options.complexMeasurements = false;
end
if options.useKinematicModel && options.regularization == 2
    error('SPKF is not compatible with the kinematic model')
end
if options.algorithm == 2 && options.complexType == 3
    error('Complex type 3 is not supported with algorithm 2')
end
if options.computeConsistency
    options.Nt = options.consistencyLength + options.stepSize + options.consistencyIter;
    %     if options.computeBayesianP && options.complexType == 3
    %         warning('Bayesian p-test is not supported with complex type 3, disabling p-test.')
    %         options.computeBayesianP = false;
    %     end
end
options.N = options.Nx * options.Ny * options.Nz;
options.NXYZ = options.N;
if ~options.use3D && options.Nz > 1
    options.N = options.Nx * options.Ny;
    DimZ = 1;
    if numel(options.m0) > options.Nt * options.Nm * options.Nz
        apu = zeros(options.Nt * options.Nm * options.Nz,1,'single');
        koko = numel(options.m0) / options.Nz;
        for kk = 1 : options.Nz
            apu(options.Nt * options.Nm * (kk-1) + 1:options.Nt * options.Nm * kk) = options.m0(koko * (kk-1) + 1:koko * (kk - 1) + options.Nt * options.Nm);
        end
        options.m0 = apu;
    end
else
    DimZ = options.Nz;
    if numel(options.m0) > options.Nt * options.Nm
        options.m0 = options.m0(1:options.Nt * options.Nm);
    end
end
options.N2 = options.Nx * options.Ny;
options.NN = options.N * options.N;
N3 = options.N;
% Increase dimension for complex type 3
if options.complexType == 3
    N3 = N3 * 2;
end
if ~isreal(options.H)
    if options.complexType ~= 3
        warning('System matrix options.H is complex, yet options.complexData is set to use real-valued data. Using only the real part!')
        options.H = real(options.H);
        options.complexS = false;
    else
        options.complexS = true;
    end
    %     if options.complexType == 3
    %         options.H = [real(options.H) -imag(options.H)];
    %         options.complexS = false;
    %     end
else
    options.complexS = false;
end
% Is H sparse?
if issparse(options.H)
    options.sparseS = true;
else
    options.sparseS = false;
end
% is Q time-varying?
if (size(options.Q,2) > 1 && ~iscell(options.Q)) || (iscell(options.Q) && length(options.Q) > 1)
    options.tvQ = true;
else
    options.tvQ = false;
end
% is R time-varying?
if (size(options.R,2) > 1 && ~iscell(options.R)) || (iscell(options.R) && length(options.R) > 1)
    options.tvR = true;
else
    options.tvR = false;
end
% Convert to single if necessary
if ~issparse(options.Q) && ~iscell(options.Q) && isa(options.Q,'double')
    options.Q = single(options.Q);
end
if ~issparse(options.R) && ~iscell(options.R) && isa(options.R,'double')
    options.R = single(options.R);
end
if isfield(options, 'F') && ~issparse(options.F) && ~iscell(options.F) && isa(options.F,'double')
    options.F = single(options.F);
end
if isfield(options, 'G') && ~issparse(options.G) && ~iscell(options.G) && isa(options.G,'double')
    options.G = single(options.G);
end
if options.algorithm == 0
    disp('Using regular Kalman filter')
elseif options.algorithm == 1
    disp('Using information filter')
elseif options.algorithm == 2
    disp('Using one step Kalman filter')
elseif options.algorithm == 3
    disp('Using ensemble Kalman filter')
elseif options.algorithm == 4
    disp('Using alternate ensemble Kalman filter')
elseif options.algorithm == 5
    disp('Using Cholesky ensemble Kalman filter')
elseif options.algorithm == 6
    disp('Using square root ensemble Kalman filter')
elseif options.algorithm == 7
    disp('Using ensemble transform Kalman filter')
elseif options.algorithm == 8
    disp('Using error subspace transform Kalman filter')
elseif options.algorithm == 9
    disp('Using dimension reduction Kalman filter')
elseif options.algorithm == 10
    disp('Using conjugate gradient Kalman filter')
elseif options.algorithm == 11
    disp('Using conjugate gradient variational Kalman filter')
elseif options.algorithm == 12
    disp('Using spectral Kalman filter')
else
    error('Invalid algorithm')
end
if options.complexType == 0
    disp('Using real-valued data (complex type 0)')
elseif options.complexType == 1
    disp('Using complex-valued data with identical covariances (complex type 1)')
elseif options.complexType == 2
    disp('Using complex-valued data with separate covariances (complex type 2)')
elseif options.complexType == 3
    disp('Using complex-valued data with split covariances (complex type 3)')
end
if options.regularization == 0
    disp('No spatial regularization selected')
elseif options.regularization == 1
    if options.Ltype == 0
        disp('Using augmented KF with 1D first order difference matrix')
    elseif options.Ltype == 1
        disp('Using augmented KF with 2D first order difference matrix')
    elseif options.Ltype == 2
        disp('Using augmented KF with 2D second order difference matrix')
    elseif options.Ltype == 3
        disp('Using augmented KF with 2D first + second order difference matrix')
    elseif options.Ltype == 4
        disp('Using augmented KF with 3D first order difference matrix')
    elseif options.Ltype == 5
        disp('Using augmented KF with custom regularization matrix')
    end
    if options.augType == 0
        disp('No weighting applied to L')
    elseif options.augType == 1
        disp('Reference image weighting of type 1 applied to L')
    elseif options.augType == 2
        disp('Reference image weighting of type 2 applied to L')
    end
elseif options.regularization == 2
    if options.Ltype == 0
        disp('Using spatial prior KF with 1D first order difference matrix')
    elseif options.Ltype == 1
        disp('Using spatial prior KF with 2D first order difference matrix')
    elseif options.Ltype == 2
        disp('Using spatial prior KF with 2D second order difference matrix')
    elseif options.Ltype == 3
        disp('Using spatial prior KF with 2D first + second order difference matrix')
    elseif options.Ltype == 4
        disp('Using spatial prior KF with 3D first order difference matrix')
    elseif options.Ltype == 5
        disp('Using spatial prior KF with custom regularization matrix')
    end
elseif options.regularization == 3
    if options.prior == 1
        if options.useAnatomical
            disp('Using denoising type 1 with reference data weighted TV-type 1')
        else
            disp('Using denoising type 1 with TV-type 1')
        end
    elseif options.prior == 2
        if options.useAnatomical
            disp('Using denoising type 1 with reference data weighted TV-type 2')
        else
            disp('Using denoising type 1 with TV-type 2')
        end
    elseif options.prior == 3
        disp('Using denoising type 1 with reference data weighted APLS')
    elseif options.prior == 4
        disp('Using denoising type 1 with SATV')
    elseif options.prior == 7
        disp('Using denoising type 1 with regularization matrix L')
    elseif options.prior == 8
        disp('Using Chambolle-Pock based TGV denoising')
    end
elseif options.regularization == 4
    disp('Using denoising type 2')
end
if options.matern
    disp('Using Matern process noise covariance')
end
if options.steadyKF && (options.algorithm == 0 || options.algorithm == 9)
    disp('Using steady state KF')
elseif options.steadyKF
    warning('Steady state is supported with algorithms 0 and 9 only. Disabling steady state.')
    options.steadyKF = false;
end
if options.useKS && (options.algorithm == 0 || options.algorithm == 1 || (options.algorithm >= 7 && options.algorithm <= 11))
    disp('Using Kalman smoother')
elseif options.useKS
    warning('Kalman smoother is used with unsupported algorithm. Disabling smoother.')
    options.useKS = false;
end
if options.computeConsistency && (options.algorithm == 0 || options.algorithm == 10)
    disp('Computing consistency tests')
elseif options.computeConsistency
    error('Consistency used with unsupported algorithm.')
end
if options.window > 1
    disp(['Sliding window KF enabled with a window length of ' num2str(options.window)])
end
if ~options.use3D && options.Nz > 1
    disp('Computing pseudo-3D estimates')
end
if options.useKinematicModel && options.regularization ~= 2 %&& options.algorithm ~= 1
    disp('Using kinematic model')
elseif options.useKinematicModel
    error('Kinematic model not supported with algorithm 1 or SPKF')
end
if options.backend == 0
    inffo = ArrayFire_OpenCL_device_info();
    k2 = strfind(inffo, 'MB');
    if options.device == 0
        k1 = strfind(inffo, '[0]');
    else
        k1 = strfind(inffo, ['-' num2str(options.device) '-']);
    end
    disp(['Computations performed with OpenCL using ', inffo(k1 + 4:k2(1)+1)])
elseif options.backend == 1
    inffo = ArrayFire_CUDA_device_info();
    k2 = strfind(inffo, 'MB');
    if options.device == 0
        k1 = strfind(inffo, '[0]');
    else
        k1 = strfind(inffo, ['-' num2str(options.device) '-']);
    end
    disp(['Computations performed with CUDA using ', inffo(k1 + 4:k2(1)+1)])
elseif options.backend == 2
    inffo = ArrayFire_CPU_device_info();
    k2 = strfind(inffo, 'GHz');
    if options.device == 0
        k1 = strfind(inffo, '[0]');
    else
        k1 = strfind(inffo, ['-' num2str(options.device) '-']);
    end
    disp(['Computations performed with CPU using ', inffo(k1 + 4:k2(1)+2)])
else
    error('Unsupported backend')
end
% if ~issparse(options.P0) && isa(options.P0,'double')
%     options.P0 = single(options.P0);
% end
% This conditional section is based on the original code by Janne
% Hakkarainen
% https://doi.org/10.1109/TCI.2019.2896527
% Dimension reduction KF
if options.algorithm == 9
    if ~options.useCustomCov
        xx = single(linspace(0,options.covDimX - 1,options.covDimX));
        yy = single(linspace(0,options.covDimY - 1,options.covDimY));
        zz = single(linspace(0,options.covDimZ - 1,options.covDimZ));
        [X,Y,Z] = meshgrid(xx,yy,zz);
        newlat = single(linspace(0,options.covDimX - 1,options.Nx));
        newlon = single(linspace(0,options.covDimY - 1,options.Ny));
        newsli = single(linspace(0,options.covDimZ - 1,options.Nz));
        [options.XX,options.YY,options.ZZ] = meshgrid(newlon,newlat,newsli);
        n = options.covDimX * options.covDimY * options.covDimZ;
        
        
        options.Sigma = zeros(n, 'single');
        for ii = 1:n
            Xd = abs(X(ii)-X);
            Yd = abs(Y(ii)-Y);
            if options.Nz > 1
                Zd = abs(Z(ii) - Z);
                d = sqrt(Xd.^2 + Yd.^2 + Zd.^2);
            else
                d = sqrt(Xd.^2 + Yd.^2);
            end
            options.Sigma(ii,:) = (options.reducedCovSigma)^2.*exp(-0.5*(d(:)./options.reducedCorLength).^2);
            %         C(ii,ii) = C(ii,ii)+tau;
        end
%         if options.Nz > 1 && options.covDimZ > 1
            if options.backend == 0
                [U, S] = OMEGASVD_OpenCL(options);
            elseif options.backend == 1
                [U, S] = OMEGASVD_CUDA(options);
            elseif options.backend == 2
                [U, S, ~] = svd(options.Sigma);
            end
            P = bsxfun(@times, S(1 : options.reducedBasisN)', U(:,1:options.reducedBasisN));
            options.Pred = zeros(options.N,options.reducedBasisN, 'single');
            
            vec = @(x) x(:);
            
            for ii=1:options.reducedBasisN
                if options.covDimZ == 1
                    options.Pred(:,ii) = vec(interp2(X,Y,reshape(P(:,ii),options.covDimX,options.covDimY),options.XX,options.YY,'linear'));
                else
                    options.Pred(:,ii) = vec(interp3(X,Y,Z, reshape(P(:,ii),options.covDimX,options.covDimY,options.covDimZ),options.XX,options.YY, options.ZZ,'linear'));
                end
            end
            options.covDimZ = 2;
            if options.useKinematicModel
                options.Pred = [options.Pred;options.Pred];
            end
            if options.complexType == 3
                options.Pred = [options.Pred;options.Pred];
            end
%         end
    else
        options.Sigma = single(options.Sigma);
    end
    options.reducedBasisN = options.reducedBasisN - 1;
end
% Matern prior
if options.matern
    rep=zeros(options.Nx,1);
    i=1;
    all_indices = cell(options.Nx*options.Ny,1);
    all_alkiot = cell(options.Nx*options.Ny,1);
    for pxCol = 1:options.Nx % fixed pixel column
        for pxRow=1:options.Ny
            for pxSlice = 1 : DimZ
                
                alkur=pxRow-options.maternDistance;
                loppur=pxRow+options.maternDistance;
                if alkur <= 0
                    alkur=1;
                end
                if loppur > options.Ny
                    loppur=options.Ny;
                end
                alkuc=pxCol-options.maternDistance;
                loppuc=pxCol+options.maternDistance;
                if alkuc <= 0
                    alkuc=1;
                end
                if loppuc > options.Nx
                    loppuc=options.Nx;
                end
                alkuS = pxSlice - options.maternDistance;
                loppuS = pxSlice + options.maternDistance;
                if alkuS <= 0
                    alkuS = 1;
                end
                if loppuS > DimZ
                    loppuS = DimZ;
                end
                r = int32(alkur:loppur);
                c = int32(alkuc:loppuc)';
                if DimZ > 1
                    z = int32(alkuS:loppuS)';
                    z = permute(z, [3 2 1]);
                    calkiot=(sqrt(single(repmat((r-pxRow).^2,length(c),1, length(z)) + repmat((c-pxCol).^2,1,length(r),length(z)) + repmat((z-pxSlice).^2,length(c),length(r), 1))));
                    calkiot = permute(calkiot, [2 1 3]);
                else
                    calkiot=(sqrt(single(repmat((r-pxRow).^2,length(c),1) + repmat(repmat((c-pxCol).^2,1,length(r)),1))))';
                end
                calkiot=calkiot(:);
                if DimZ > 1
                    c_indices=repmat(r',length(c),length(z)) + repmat(repelem(c*options.Nx,length(r)),1,length(z))-options.Nx + repmat((z(:)' - 1) * options.N2, length(c) * length(r), 1);
                else
                    c_indices=(repmat(repmat(r',length(c),1),1) + repmat(repelem(c*options.Nx,length(r)),1)-options.Nx);
                end
                c_indices=c_indices(calkiot>0);
                calkiot=calkiot(calkiot>0);
                all_indices{i}=c_indices;
                all_alkiot{i}=calkiot;
                rep(i)=length(c_indices);
                i = i+1;
            end
        end
    end
    all_indices = cell2mat(all_indices);
    all_alkiot = cell2mat(all_alkiot);
    %     end
    all_indices(all_indices==0)=[];
    all_alkiot(all_alkiot==0)=[];
    RR=sparse(repelem(double(1:options.Nx*options.Ny),rep)',double(all_indices),double(all_alkiot));
    if options.maternNu == 5/2
        if issparse(RR)
            CW=options.maternSigma*((spones(RR)+speye(size(RR)))+(sqrt(5)*RR)/options.maternl+(5*RR.^2)/(3*options.maternl^2)).*exp(-(sqrt(5)*RR)/options.maternl);
        else
            CW=options.maternSigma*(ones(size(RR))+(sqrt(5)*RR)/options.maternl+(5*RR.^2)/(3*options.maternl^2)).*exp(-(sqrt(5)*RR)/options.maternl);
        end
    elseif options.maternNu == 3/2
        if issparse(RR)
            CW=options.maternSigma*((spones(RR)+speye(size(RR)))+(sqrt(3)*RR)/options.maternl).*exp(-(sqrt(3)*RR)/options.maternl);
        else
            CW=options.maternSigma*(ones(size(RR))+(sqrt(3)*RR)/options.maternl).*exp(-(sqrt(3)*RR)/options.maternl);
        end
    elseif options.maternNu == 1/2
        if issparse(RR)
            CW=options.maternSigma^2*(spones(RR)+speye(size(RR))).*exp(-RR/options.maternl);
        else
            CW=options.maternSigma^2*exp(-RR/options.maternl);
        end
    elseif options.maternNu == Inf
        if issparse(RR)
            CW=options.maternSigma^2*(spones(RR)+speye(size(RR))).*exp(-RR.^2/(2*options.maternl^2));
        else
            CW=options.maternSigma^2.*exp(-RR.^2/(2*options.maternl^2));
        end
    else
        error('Invalid nu value')
    end
    CW=2*options.maternLambda*CW;
    options.Q =(1/(2*options.maternLambda))*(CW-exp(-2*options.maternLambda*options.maternDeltaT)*CW);
    %     QQ=(single(full(QQ)));
end
% Construct Q if necessary
if size(options.Q,1) == 1 && size(options.Q,2) == 1 && ~iscell(options.Q)
    if isreal(options.Q) && options.complexType > 1
        options.Q = complex(ones(options.NXYZ,1,'single'),ones(options.NXYZ,1,'single')) * real(options.Q);
    elseif options.complexType > 1
        options.Q = complex(ones(options.NXYZ,1,'single') .* real(options.Q),ones(options.NXYZ,1,'single') .* imag(options.Q));
    else
        options.Q = ones(options.NXYZ,1,'single') * real(options.Q);
    end
elseif size(options.Q,1) == 1 && size(options.Q,2) > 1 && ~iscell(options.Q)
    options.Q = single(reshape(options.Q, [], 1));
    if options.complexType > 1 && isreal(options.Q)
        options.Q = complex(options.Q,options.Q);
    end
elseif ~iscell(options.Q) && ~issparse(options.Q)
    options.Q = single(options.Q);
    if options.complexType > 1 && isreal(options.Q)
        options.Q = complex(options.Q,options.Q);
    end
elseif ~iscell(options.Q) && issparse(options.Q)
    if options.complexType > 1 && isreal(options.Q)
        options.Q = complex(options.Q,options.Q);
    end
end
% Construct R if necessary
if size(options.R,1) == 1 && size(options.R,2) == 1 && ~iscell(options.R)
    if isreal(options.R) && options.complexType > 1
        options.R = complex(ones(options.Nm * options.window,1,'single'),ones(options.Nm * options.window,1,'single')) * real(options.R);
    elseif options.complexType > 1
        options.R = complex(ones(options.Nm * options.window,1,'single') .* real(options.R),ones(options.Nm * options.window,1,'single') .* imag(options.R));
    else
        options.R = ones(options.Nm * options.window,1,'single') * real(options.R);
    end
elseif size(options.R,1) == 1 && size(options.R,2) > 1 && ~iscell(options.R)
    options.R = reshape(options.R, [], 1);
elseif ~iscell(options.R) && ~issparse(options.R)
    options.R = single(options.R);
    if options.complexType > 1 && isreal(options.R)
        options.R = complex(options.R,options.R);
    end
end
% Construct the initial a priori error covariance
if size(options.P0,1) == 1
    if isreal(options.P0) && options.complexType > 1
        if options.algorithm == 1
            options.P0 = complex(ones(options.N,1,'single'),ones(options.N,1,'single')) ./ real(options.P0);
        elseif options.algorithm == 10 || options.algorithm == 11
            options.P0 = complex(ones(options.N,options.cgIter,'single'),ones(options.N,options.cgIter,'single')) .* real(options.P0);
        else
            options.P0 = complex(ones(options.N,1,'single'),ones(options.N,1,'single')) .* real(options.P0);
        end
    elseif options.complexType > 1
        if options.algorithm == 1
            options.P0 = complex(ones(options.N,1,'single'),ones(options.N,1,'single')) ./ [real(options.P0);imag(options.P0)];
        elseif options.algorithm == 10 || options.algorithm == 11
            options.P0 = complex(ones(options.N,options.cgIter,'single'),ones(options.N,options.cgIter,'single')) .* [real(options.P0);imag(options.P0)];
        else
            options.P0 = complex(ones(options.N,1,'single'),ones(options.N,1,'single')) .* [real(options.P0);imag(options.P0)];
        end
    else
        if options.algorithm == 1
            options.P0 = ones(options.N,1,'single') ./ options.P0;
        elseif options.algorithm == 10 || options.algorithm == 11
            if options.algorithm && options.complexType == 1
                options.P0 = complex(ones(options.N,options.cgIter,'single') .* options.P0, ones(options.N,options.cgIter,'single') .* options.P0);
            else
                options.P0 = ones(options.N,options.cgIter,'single') .* options.P0;
            end
        else
            options.P0 = ones(options.N,1,'single') .* options.P0;
        end
    end
    if options.useKinematicModel
        options.P0 = repmat(options.P0, 2, 1);
    end
else
    if options.algorithm == 1
        options.P0 = inverse(single(options.P0));
    else
        options.P0 = single(options.P0);
    end
end
options.m0 = single(options.m0);
% if numel(options.m0) == size(options.m0,1) && options.complexType == 3
%     options.m0 = reshape(options.m0, options.Nm, []);
% end
% if numel(options.m0) / options.Nm ~= options.Nt
%     options.m0 = options.m0(1 : options.Nm * options.Nt * options.window);
% end
% options.m0 = reshape(options.m0, options.Nm * options.window, options.Nt);
if size(options.H, 2) == options.N || (size(options.H,2) == (options.Nx * options.Ny) && options.use3D == false)% && issparse(options.H)
    options.H = options.H.';
end
if ~isreal(options.referenceImage) && options.complexRef && ((options.regularization == 1 && options.augType > 0) || ...
        (options.regularization == 3 && options.useAnatomical) || (options.regularization == 3 && options.prior == 7))
    options.complexRef = true;
else
    options.complexRef = false;
end
extseq = single(0);
options.tau = (extseq);
if options.regularization == 3 && options.prior > 0 && options.prior < 5 && options.useAnatomical
    %     if options.complexRef
    %         S = assembleS(options.referenceImage,options.C,options.Ny,options.Nx,options.Nz);
    %     else
    S = assembleS(real(options.referenceImage),options.C,options.Ny,options.Nx,options.Nz);
    %     end
    
    options.s1 = single(S(1:3:end,1));
    options.s2 = single(S(1:3:end,2));
    options.s3 = single(S(1:3:end,3));
    options.s4 = single(S(2:3:end,1));
    options.s5 = single(S(2:3:end,2));
    options.s6 = single(S(2:3:end,3));
    options.s7 = single(S(3:3:end,1));
    options.s8 = single(S(3:3:end,2));
    options.s9 = single(S(3:3:end,3));
    if options.complexRef
        S = assembleS(imag(options.referenceImage),options.CI,options.Ny,options.Nx,options.Nz);
        options.s1 = [options.s1;(single(S(1:3:end,1)))];
        options.s2 = [options.s2;(single(S(1:3:end,2)))];
        options.s3 = [options.s3;(single(S(1:3:end,3)))];
        options.s4 = [options.s4;(single(S(2:3:end,1)))];
        options.s5 = [options.s5;(single(S(2:3:end,2)))];
        options.s6 = [options.s6;(single(S(2:3:end,3)))];
        options.s7 = [options.s7;(single(S(3:3:end,1)))];
        options.s8 = [options.s8;(single(S(3:3:end,2)))];
        options.s9 = [options.s9;(single(S(3:3:end,3)))];
    end
end
if options.useAnatomical
    options.referenceImage = single(options.referenceImage);
end
% Regularization matrix L
if options.regularization == 1 || options.regularization == 2 || options.regularization == 4 || (options.regularization == 3 && options.prior == 7)
    if options.Ltype == 0
        options.colIndL = zeros(options.N * 2 - 1,1,'int32');
        options.colIndL(1 : 2 : end) = int32(0 : options.N - 1)';
        options.colIndL(2 : 2 : end) = int32(1 : options.N - 1)';
        options.rowIndL = [int32(0);int32(2 : 2 : options.N * 2 - 1)';options.N * 2 - 1];
        options.valL = ones(options.N * 2 - 1, 1, 'single');
        options.valL(1 : 2 : end) = options.valL(1 : 2 : end) * single(-1);
        if ~issparse(options.H) || options.regularization == 2
            rows = (1 : options.N)';
            options.rowIndL = diff(options.rowIndL);
            rows = repelem(rows, options.rowIndL);
            options.L = full(sparse(rows, double(options.colIndL + 1), double(options.valL)));
        end
    elseif options.Ltype == 1
        options.colIndL = zeros(options.N * 3 - options.Nx - options.Ny,1,'int32');
        options.valL = ones(options.N * 3 - options.Nx - options.Ny,1, 'single');
        options.rowIndL = zeros(options.N + 1,1,'int32');
        for kk = 1 : options.Nx
            if kk < options.Nx
                options.colIndL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1)) = int32((kk - 1)*options.Ny : kk*options.Ny - 2)';
                options.colIndL(2 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1) + 1) = int32((kk - 1)*options.Ny + 1 : kk*options.Ny - 2 + 1)';
                options.colIndL(3 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1) + 2) = int32((kk)*options.Nx : (kk + 1)*options.Nx - 2)';
                options.valL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1)) = -2;
                options.rowIndL(2 + (kk - 1) * (options.Ny): kk * options.Ny) = int32(3 + (kk - 1) * options.Ny * 3 - (kk - 1): 3:3 + (kk) * options.Ny * 3 - 3*2 - (kk - 1));
            else
                options.colIndL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 2 : end-1) = int32((kk - 1)*options.Ny : kk*options.Ny - 2)';
                options.valL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 2 : end-1) = -1;
                options.colIndL(options.N * 3 - options.Ny * 4 + 3: 2 : end) = int32(options.N - options.Ny + 1 : options.N - 1)';
            end
        end
        options.valL(options.Ny * 3 - 2: options.Ny * 3 - 1 : end - options.Nx - 1) = -1;
        options.colIndL(options.Ny * 3 - 2: options.Ny * 3 - 1 : end - options.Nx - 1) = int32(options.Ny - 1 : options.Ny : options.N - options.Nx)';
        options.colIndL(end) = options.N - 1;
        options.rowIndL(options.Ny + 1 : options.Ny : end - options.Nx) = options.rowIndL(options.Ny : options.Ny : end - options.Nx - 1) + 2;
        options.rowIndL(end - options.Nx + 1 : end) = options.rowIndL(end - options.Nx) + 2:2:options.rowIndL(end - options.Nx) + 2 * ...
            length(options.rowIndL(end - options.Nx + 1 : end));
        options.rowIndL(end) = options.rowIndL(end - 1) + 1;
        options.valL(end) = 1;
        options.colIndL(options.Ny * 3 - 1: options.Ny * 3 - 1 : end - options.Nx) = int32(options.Ny * 2 - 1 : options.Ny : options.N)';
        if options.regularization == 2 || options.augType > 0
            apu = 1 : length(options.rowIndL)-1;
            apu2 = diff(options.rowIndL);
            options.rowIndL = repelem(apu, apu2);
            options.L = sparse(double(options.rowIndL),double(options.colIndL) + 1,double(options.valL),options.N,options.N);
            if ~issparse(options.H)
                options.L = full(options.L);
            end
            clear apu apu2
        elseif ~issparse(options.H)
            rows = (1 : options.N)';
            options.rowIndL = diff(options.rowIndL);
            rows = repelem(rows, options.rowIndL);
            options.L = full(sparse(rows, double(options.colIndL + 1), double(options.valL)));
        end
    elseif options.Ltype == 2 || options.Ltype == 3
        if options.Ltype == 3
            
            colIndL = zeros(options.N * 3 - options.Nx - options.Ny,1,'int32');
            valL = ones(options.N * 3 - options.Nx - options.Ny,1, 'single');
            rowIndL = zeros(options.N + 1,1,'int32');
            for kk = 1 : options.Nx
                if kk < options.Nx
                    colIndL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1)) = int32((kk - 1)*options.Ny : kk*options.Ny - 2)';
                    colIndL(2 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1) + 1) = int32((kk - 1)*options.Ny + 1 : kk*options.Ny - 2 + 1)';
                    colIndL(3 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1) + 2) = int32((kk)*options.Nx : (kk + 1)*options.Nx - 2)';
                    valL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 3 : kk * options.Ny * 3 - 3 - (kk - 1)) = -2;
                    rowIndL(2 + (kk - 1) * (options.Ny): kk * options.Ny) = int32(3 + (kk - 1) * options.Ny * 3 - (kk - 1): 3:3 + (kk) * options.Ny * 3 - 3*2 - (kk - 1));
                else
                    colIndL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 2 : end-1) = int32((kk - 1)*options.Ny : kk*options.Ny - 2)';
                    valL(1 + (kk - 1) * (options.Ny * 3) - (kk - 1): 2 : end-1) = -1;
                    colIndL(options.N * 3 - options.Ny * 4 + 3: 2 : end) = int32(options.N - options.Ny + 1 : options.N - 1)';
                end
            end
            valL(options.Ny * 3 - 2: options.Ny * 3 - 1 : end - options.Nx - 1) = -1;
            colIndL(options.Ny * 3 - 2: options.Ny * 3 - 1 : end - options.Nx - 1) = int32(options.Ny - 1 : options.Ny : options.N - options.Nx)';
            colIndL(end) = options.N - 1;
            rowIndL(options.Ny + 1 : options.Ny : end - options.Nx) = rowIndL(options.Ny : options.Ny : end - options.Nx - 1) + 2;
            rowIndL(end - options.Nx + 1 : end) = rowIndL(end - options.Nx) + 2:2:rowIndL(end - options.Nx) + 2 * ...
                length(rowIndL(end - options.Nx + 1 : end));
            rowIndL(end) = rowIndL(end - 1) + 1;
            valL(end) = 1;
            colIndL(options.Ny * 3 - 1: options.Ny * 3 - 1 : end - options.Nx) = int32(options.Ny * 2 - 1 : options.Ny : options.N)';
            rows = (1 : options.N)';
            rowIndL = diff(rowIndL);
            rows = repelem(rows, rowIndL);
            options.L = full(sparse(rows, double(colIndL + 1), double(valL)));
            options.L = spdiags([ones(options.N2,1)*-2,ones(options.N2,1),ones(options.N2,1)],[0,1,options.Ny], options.N2, options.N2);
            options.L(options.N2*(options.Nx - 1)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(options.L)) = -1;
            options.L(options.N2*(options.Nx-1)*(options.Ny)+options.N2-options.Nx + 1:options.N2+1:numel(options.L)) = -1;
            options.L(options.N2*(options.Nx)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(options.L)) = 0;
            options.L(end) = 1;
            L = spdiags([ones(options.N2,1)*2,ones(options.N2,1)*-2,ones(options.N2,1),ones(options.N2,1)*-2,ones(options.N2,1)],[0,1,2,options.Ny, options.Ny*2], options.N2, options.N2);
            L(options.N2*(options.Nx - 1)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(L)) = 1;
            L(options.N2*(options.Nx)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(L)) = 0;
            L(options.N2*(options.Nx + 1)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(L)) = 0;
            L(options.N2*(options.Nx - 1)+options.Nx - 1:options.N2*(options.Nx)+options.Nx:numel(L)) = -1;
            L(options.N2*(options.Nx)+options.Nx - 1:options.N2*(options.Nx)+options.Nx:numel(L)) = 0;
            L(options.N2*(options.Nx-1)*(options.Ny)+options.N2-options.Nx + 1:options.N2+1:numel(L)) = 1;
            L(options.N2*(options.Nx-1)*(options.Ny)+options.N2-options.Nx*2 + 1:options.N2+1:numel(L)) = -1;
            options.L = options.L + L .* options.Lscale;
        else
            options.L = spdiags([ones(options.N2,1)*2,ones(options.N2,1)*-2,ones(options.N2,1),ones(options.N2,1)*-2,ones(options.N2,1)],[0,1,2,options.Ny, options.Ny*2], options.N2, options.N2);
            options.L(options.N2*(options.Nx - 1)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(options.L)) = 1;
            options.L(options.N2*(options.Nx)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(options.L)) = 0;
            options.L(options.N2*(options.Nx + 1)+options.Nx:options.N2*(options.Nx)+options.Nx:numel(options.L)) = 0;
            options.L(options.N2*(options.Nx - 1)+options.Nx - 1:options.N2*(options.Nx)+options.Nx:numel(options.L)) = -1;
            options.L(options.N2*(options.Nx)+options.Nx - 1:options.N2*(options.Nx)+options.Nx:numel(options.L)) = 0;
            options.L(options.N2*(options.Nx-1)*(options.Ny)+options.N2-options.Nx + 1:options.N2+1:numel(options.L)) = 1;
            options.L(options.N2*(options.Nx-1)*(options.Ny)+options.N2-options.Nx*2 + 1:options.N2+1:numel(options.L)) = -1;
        end
        if ~issparse(options.H)
            options.L = full(options.L);
        end
    elseif options.Ltype == 4
        options.L = spdiags([ones(options.N,1)*-3,ones(options.N,1),ones(options.N,1),ones(options.N,1)],[0,1,options.Ny, options.N2],options.N, options.N);
        options.L(options.N*(options.Ny - 1)+options.Ny:options.N*(options.Nx)+options.Nx:numel(options.L)) = -2;
        for kk = 1 : options.Nz - 1
            options.L(options.N*options.N2 * kk + options.N2 * kk + 1 - options.Nx : options.N*(options.Nx)+options.Nx:options.N*options.N2 * kk + options.N2 * kk) = 0;
            options.L(options.N*(options.N2 * kk - options.Nx) + options.N2 * kk - options.Nx + 1 : options.N*(options.Nx)+options.Nx: options.N*(options.N2 * kk) - options.N2 * kk) = -2;
        end
        options.L(options.N*options.N2 * (options.Nz - 1) + options.N2 * (options.Nz - 1) + options.N*(options.Ny - 1)+options.Ny:options.N*(options.Nx)+options.Nx:numel(options.L)) = -1;
        options.L(options.N*options.N2 * (options.Nz - 1) + options.N2 * (options.Nz - 1) + 1:options.N+1:numel(options.L)) = -2;
        options.L(options.N*(options.N - options.Nx) + options.N2*options.Nz - options.Nx + 1:options.N+1:numel(options.L)) = -1;
        if ~issparse(options.H)
            options.L = full(options.L);
        end
    end
end
% Weighting for the L matrix
if (options.regularization == 1 || options.regularization == 2 || options.regularization == 4 || (options.regularization == 3 && options.prior == 7)) && options.augType > 0
    xRef = reshape(options.referenceImage,options.Nx,options.Ny,options.Nz);
    xRef = real(xRef);
    if options.augType == 1
        kx = (abs(exp(-cat(2,abs(diff(xRef,1,2)),zeros(options.Nx,1,options.Nz))/options.C)));
        ky = (abs(exp(-cat(1,abs(diff(xRef)),zeros(1,options.Ny,options.Nz))/options.C)));
        if options.Ltype == 2 || options.Ltype == 3
            kx2 = (abs(exp(-cat(2,abs(diff(xRef,2,2)),zeros(options.Nx,1,options.Nz))/options.C)));
            ky2 = (abs(exp(-cat(1,abs(diff(xRef,2,1)),zeros(1,options.Ny,options.Nz))/options.C)));
        elseif options.Ltype == 4
            kz = (abs(exp(-cat(3,abs(diff(xRef,1,3)), zeros(options.Nx,options.Ny,1))/options.C)));
        end
    elseif options.augType == 2
        %         nbr = AssembleENBR(single(options.Nx),single(options.Ny));
        %         nbr.edx=single(nbr.edx);
        %         nbr.edy=single(nbr.edy);
        %         extseq = single(0);
        S = assembleS((xRef),(options.C),options.Ny,options.Nx,options.Nz);
        if options.Nz == 1
            S(3:3:end,:) = [];
            S = S(:,1:2);
            kx=(single(S(1:2:end,1)));
            ky=(single(S(2:2:end,2)));
        else
            kx=(single(S(1:3:end,1)));
            ky=(single(S(2:3:end,2)));
            kz = single(S(3:3:end,3));
        end
    end
    if options.Ltype == 0
        kappa = spdiags([ones(options.N,1),ky(:)],[0,1], options.N, options.N);
    elseif options.Ltype == 1
        kappa = spdiags([ones(options.N,1),ky(:),kx(:)],[0,1, options.Nx], options.N, options.N);
    elseif options.Ltype == 2 || options.Ltype == 3
        kappa = spdiags([ones(options.N,1),ky(:), [ky2(:);ones(options.Ny,1)],kx(:),[kx2(:);ones(options.Nx,1)]],[0,1, 2, options.Nx, options.Nx + 1], options.N, options.N);
    elseif options.Ltype == 4
        kappa = spdiags([ones(options.N,1),ky(:),kx(:),kz(:)],[0,1, options.Nx, options.N2], options.N, options.N);
    else
        kappa = 1;
    end
    if options.regularization == 1
        if isfield(options,'L')
            options.L = options.beta .* kappa .* options.L;
        else
            options.valL = options.beta .* nonzeros(kappa) .* options.valL;
        end
    elseif options.regularization == 4 || (options.regularization == 3 && options.prior == 7)
        if isfield(options,'L')
            options.L = kappa .* options.L;
        else
            options.valL = nonzeros(kappa) .* options.valL;
        end
    else
        if options.complexType > 1
            options.Qi = options.Q;
        end
        if options.tvQ
            options.F = cell(options.Nt,1);
            if ~iscell(options.Q) && ~issparse(options.Q)
                apuQ = real(options.Q);
                options.Q = cell(options.Nt,1);
            end
            for kk = 1 : options.Nt
                if iscell(options.Q) && issparse(options.Q{kk})
                    options.F{kk} = full(inv(speye(options.N,options.N) + real(options.Q{kk}) .* options.beta .* (kappa .* options.L)' * (kappa .* options.L)));
                elseif iscell(options.Q)
                    options.F{kk} = full(inv(speye(options.N,options.N) + bsxfun(@times, double(real(options.Q{kk})), options.beta .* (kappa .* options.L)' * (kappa .* options.L))));
                else
                    options.F{kk} = full(inv(speye(options.N,options.N) + bsxfun(@times, double(apuQ(:,kk)), options.beta .* (kappa .* options.L)' * (kappa .* options.L))));
                end
                if ~options.complexRef && options.complexType > 1
                    if iscell(options.Q) && issparse(options.Q{kk})
                        options.F{kk} = complex(options.F{kk}, full(inv(speye(options.N,options.N) + imag(options.Q{kk}) .* options.betaI .* (kappa .* options.L)' * (kappa .* options.L))));
                    elseif iscell(options.Q)
                        options.F{kk} = complex(options.F{kk}, full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(options.Q{kk})), options.betaI .* (kappa .* options.L)' * (kappa .* options.L)))));
                    else
                        options.F{kk} = complex(options.F{kk}, full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(options.Q(:,kk))), options.betaI .* (kappa .* options.L)' * (kappa .* options.L)))));
                    end
                end
                options.F{kk}(abs(options.F{kk}) < 1e-12) = 0;
                options.F{kk} = sparse(options.F{kk});
                if options.complexType <= 2
                    if iscell(options.Q) && issparse(options.Q{kk})
                        options.Q{kk} = real(options.Q{kk}) * real(options.F{kk});
                    elseif iscell(options.Q)
                        options.Q{kk} = (speye(options.N,options.N) .* double(real(options.Q{kk}))) * real(options.F{kk});
                    else
                        options.Q{kk} = (speye(options.N,options.N) .* double(apuQ(:,kk))) * real(options.F{kk});
                    end
                    if ~options.complexRef && options.complexType > 1
                        if iscell(options.Qi) && issparse(options.Qi{kk})
                            options.Q{kk} = complex(options.Q{kk},imag(options.Qi{kk}) * imag(options.F{kk}));
                        elseif iscell(options.Qi)
                            options.Q{kk} = complex(options.Q{kk}, (speye(options.N,options.N) .* double(imag(options.Qi{kk}))) * imag(options.F{kk}));
                        else
                            options.Q{kk} = complex(options.Q{kk}, (speye(options.N,options.N) .* double(imag(options.Qi(:,kk)))) * imag(options.F{kk}));
                        end
                    end
                end
            end
        else
            if issparse(options.Q)
                options.F = full(inv(speye(options.N,options.N) + real(options.Q) .* options.beta .* (kappa .* options.L)' * (kappa .* options.L)));
            else
                options.F = full(inv(speye(options.N,options.N) + bsxfun(@times, double(real(options.Q)), options.beta .* (kappa .* options.L)' * (kappa .* options.L))));
            end
            if ~options.complexRef && options.complexType > 1
                if issparse(options.Q)
                    options.F = complex(options.F,full(inv(speye(options.N,options.N) + imag(options.Q) .* options.betaI .* (kappa .* options.L)' * (kappa .* options.L))));
                else
                    options.F = complex(options.F,full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(options.Q)), options.betaI .* (kappa .* options.L)' * (kappa .* options.L)))));
                end
            end
            options.F(abs(options.F) < 1e-12) = 0;
            options.F = sparse(options.F);
            if options.complexType <= 2
                if issparse(options.Q)
                    options.Q = real(options.Q) * real(options.F);
                else
                    options.Q = (speye(options.N,options.N) .* double(real(options.Q))) .* real(options.F);
                end
                if ~options.complexRef && options.complexType > 1
                    if issparse(options.Qi)
                        options.Q = complex(options.Q, imag(options.Qi) * imag(options.F));
                    else
                        options.Q = complex(options.Q, (speye(options.N,options.N) .* double(imag(options.Qi))) * imag(options.F));
                    end
                end
            end
        end
    end
    if options.complexRef
        xRef = reshape(options.referenceImage,options.Nx,options.Ny,options.Nz);
        xRef = imag(xRef);
        if options.augType == 1
            kx=(abs(exp(-cat(2,abs(diff(xRef,1,2)),zeros(options.Nx,1,options.Nz))/options.C)));
            ky=(abs(exp(-cat(1,abs(diff(xRef)),zeros(1,options.Ny,options.Nz))/options.C)));
            if options.Ltype == 2 || options.Ltype == 3
                kx2=(abs(exp(-cat(2,abs(diff(xRef,2,2)),zeros(options.Nx,1,options.Nz))/options.C)));
                ky2=(abs(exp(-cat(1,abs(diff(xRef,2,1)),zeros(1,options.Ny,options.Nz))/options.C)));
            elseif options.Ltype == 4
                kz = (abs(exp(-cat(3,abs(diff(xRef,1,3)), zeros(options.Nx,options.Ny,1))/options.C)));
            end
        elseif options.augType == 2
            S = assembleS((xRef),(options.C),options.Ny,options.Nx,options.Nz);
            if options.Nz == 1
                S(3:3:end,:) = [];
                S = S(:,1:2);
                kx=(single(S(1:2:end,1)));
                ky=(single(S(2:2:end,2)));
            else
                kx=(single(S(1:3:end,1)));
                ky=(single(S(2:3:end,2)));
                kz = single(S(3:3:end,3));
            end
        end
        if options.Ltype == 0
            kappa = spdiags([ones(options.N,1),ky(:)],[0,1], options.N, options.N);
        elseif options.Ltype == 1
            kappa = spdiags([ones(options.N,1),ky(:),kx(:)],[0,1, options.Nx], options.N, options.N);
        elseif options.Ltype == 2 || options.Ltype == 3
            kappa = spdiags([ones(options.N,1),ky(:), [ky2(:);ones(options.Ny,1)],kx(:),[kx2(:);ones(options.Nx,1)]],[0,1, 2, options.Nx, options.Nx + 1], options.N, options.N);
        elseif options.Ltype == 4
            kappa = spdiags([ones(options.N,1),ky(:),kx(:),kz(:)],[0,1, options.Nx, options.N2], options.N, options.N);
        else
            kappa = 1;
        end
        if options.algorithm == 1
            if isfield(options,'L')
                options.Li = options.betaI .* kappa .* options.L;
            else
                options.valLi = options.betaI .* nonzeros(kappa) .* options.valL;
            end
        elseif options.regularization == 4 || (options.regularization == 3 && options.prior == 7)
            if isfield(options,'L')
                options.Li = kappa .* options.L;
            else
                options.valLi = nonzeros(kappa) .* options.valL;
            end
        else
            if options.complexType > 1
                if options.tvQ
                    for kk = 1 : options.Nt
                        if iscell(options.Qi) && issparse(options.Qi{kk})
                            tempF = full(inv(speye(options.N,options.N) + imag(options.Qi{kk}) .* options.betaI .* (kappa .* options.L)' * (kappa .* options.L)));
                        elseif iscell(options.Qi)
                            tempF = full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(options.Qi{kk})), options.betaI .* (kappa .* options.L)' * (kappa .* options.L))));
                        else
                            tempF = full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(options.Qi(:,kk))), options.betaI .* (kappa .* options.L)' * (kappa .* options.L))));
                        end
                        tempF(abs(tempF) < 1e-12) = 0;
                        tempF = sparse(tempF);
                        options.F{kk} = complex(options.F{kk}, tempF);
                        if options.complexType <= 2
                            if iscell(options.Qi) && issparse(options.Qi{kk})
                                options.Q{kk} = complex(options.Q{kk},imag(options.Qi{kk}) * imag(options.F{kk}));
                            elseif iscell(options.Qi)
                                options.Q{kk} = complex(options.Q{kk}, (speye(options.N,options.N) .* double(imag(options.Qi{kk}))) * imag(options.F{kk}));
                            else
                                options.Q{kk} = complex(options.Q{kk}, (speye(options.N,options.N) .* double(imag(options.Qi(:,kk)))) * imag(options.F{kk}));
                            end
                        end
                    end
                else
                    if issparse(options.Qi)
                        tempF = full(inv(speye(options.N,options.N) + imag(options.Qi) * options.betaI .* (kappa .* options.L)' * (kappa .* options.L)));
                    else
                        tempF = full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(options.Qi)), options.betaI .* (kappa .* options.L)' * (kappa .* options.L))));
                    end
                    tempF(abs(tempF) < 1e-12) = 0;
                    tempF = sparse(tempF);
                    options.F = complex(options.F, tempF);
                    if options.complexType <= 2
                        if issparse(options.Qi)
                            options.Q = complex(options.Q, imag(options.Qi) * options.F);
                        else
                            options.Q = complex(options.Q, (speye(options.N,options.N) .* double(imag(options.Qi))) .* options.F);
                        end
                    end
                end
            end
        end
    end
    % No weighting for L
elseif (options.regularization == 1 || options.regularization == 2)
    if options.regularization == 1
        if isfield(options,'L')
            options.L = options.L * options.beta;
        else
            options.valL = options.valL * options.beta;
        end
    else
        if options.complexType > 1
            Qi = options.Q;
        end
        if options.tvQ
            options.F = cell(options.Nt,1);
            if ~iscell(options.Q) && ~issparse(options.Q)
                apuQ = real(options.Q);
                options.Q = cell(options.Nt,1);
            end
            for kk = 1 : options.Nt
                if iscell(options.Q) && issparse(options.Q{kk})
                    options.F{kk} = full(inv(speye(options.N,options.N) + real(options.Q{kk}) * (options.beta .* (options.L)' * (options.L))));
                elseif iscell(options.Q)
                    options.F{kk} = full(inv(speye(options.N,options.N) + bsxfun(@times, double(real(options.Q{kk})), options.beta .* (options.L)' * (options.L))));
                else
                    options.F{kk} = full(inv(speye(options.N,options.N) + bsxfun(@times, double(apuQ(:,kk)), options.beta .* (options.L)' * (options.L))));
                end
                options.F{kk}(abs(options.F{kk}) < 1e-12) = 0;
                options.F{kk} = sparse(options.F{kk});
                if options.complexType <= 2
                    if iscell(options.Q) && issparse(options.Q{kk})
                        options.Q{kk} = real(options.Q{kk}) * options.F{kk};
                    elseif iscell(options.Q)
                        options.Q{kk} = (speye(options.N,options.N) .* double(real(options.Q{kk}))) * options.F{kk};
                    else
                        options.Q{kk} = (speye(options.N,options.N) .* double(apuQ(:,kk))) * options.F{kk};
                    end
                end
            end
        else
            if issparse(options.Q)
                options.F = full(inv(speye(options.N,options.N) + real(options.Q) * (options.beta .* (options.L)' * (options.L))));
            else
                options.F = full(inv(speye(options.N,options.N) + bsxfun(@times, double(real(options.Q)), options.beta .* (options.L)' * (options.L))));
            end
            options.F(abs(options.F) < 1e-12) = 0;
            options.F = sparse(options.F);
            if options.complexType <= 2
                if issparse(options.Q)
                    options.Q = real(options.Q) * options.F;
                else
                    options.Q = spdiags(double(real(options.Q)),0, options.N, options.N) * options.F;
                end
            end
        end
        if options.complexType > 1
            if options.tvQ
                for kk = 1 : options.Nt
                    if iscell(Qi) && issparse(Qi{kk})
                        tempF = full(inv(speye(options.N,options.N) + imag(Qi{kk}) * (options.betaI .* (options.L)' * (options.L))));
                    elseif iscell(Qi)
                        tempF = full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(Qi{kk})), options.betaI .* (options.L)' * (options.L))));
                    else
                        tempF = full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(Qi(:,kk))), options.betaI .* (options.L)' * (options.L))));
                    end
                    tempF(abs(tempF) < 1e-12) = 0;
                    tempF = sparse(tempF);
                    options.F{kk} = complex(options.F{kk}, tempF);
                    if options.complexType <= 2
                        if iscell(Qi) && issparse(Qi{kk})
                            options.Q{kk} = complex(options.Q{kk},imag(Qi{kk}) * tempF);
                        elseif iscell(Qi)
                            options.Q{kk} = complex(options.Q{kk},(speye(options.N,options.N) .* double(imag(Qi{kk}))) * tempF);
                        else
                            options.Q{kk} = complex(options.Q{kk},(speye(options.N,options.N) .* double(imag(Qi(:,kk)))) * tempF);
                        end
                    end
                end
            else
                if issparse(Qi)
                    tempF = full(inv(speye(options.N,options.N) + imag(Qi) * (options.betaI .* (options.L)' * (options.L))));
                else
                    tempF = full(inv(speye(options.N,options.N) + bsxfun(@times, double(imag(Qi)), options.betaI .* (options.L)' * (options.L))));
                end
                tempF(abs(tempF) < 1e-12) = 0;
                tempF = sparse(tempF);
                if iscell(options.F)
                    options.F{1} = complex(options.F{1}, tempF);
                else
                    options.F = complex(options.F, tempF);
                end
                if options.complexType <= 2
                    if issparse(Qi)
                        options.Q = complex(options.Q, (imag(Qi)) * tempF);
                    else
                        options.Q = complex(options.Q,spdiags(double(imag(Qi)),0, options.N, options.N) * tempF);
                    end
                end
            end
        end
    end
end
% if issparse(options.H) && options.regularization == 1 && isfield(options,'L')
%     options.L = options.L';
% end
% Kinematic model
if options.useKinematicModel
    temp = spdiags([ones(options.N,1) *.5 * options.deltaT^2, ones(options.N,1) * options.deltaT], [0, -options.N], options.N * 2, options.N);
    if ~options.tvQ && issparse(options.Q)
        %     options.F = spdiags([ones(options.N*2,1),ones(options.N*2,1) * options.deltaT], [0,options.N], options.N*2, options.N*2);
        %         temp = spdiags([ones(options.N,1) *.5 * options.deltaT^2,ones(options.N,1) * options.deltaT], [0, -options.N], options.N * 2, options.N);
        %     if issparse(options.kQ)
        if options.complexType > 1
            options.Q = complex((temp * real(options.Q) * temp'), (temp * imag(options.Q) * temp'));
        else
            options.Q = (temp * options.Q * temp');
        end
        %     else
        %         options.Q = (bsxfun(@times, temp, double(options.Q)') * temp')';
        %     end
    elseif options.tvQ && issparse(options.Q)
        %         temp = spdiags([ones(options.N,1) *.5 * options.deltaT^2,ones(options.N,1) * options.deltaT], [0, -options.N], options.N * 2, options.N);
        for kk = 1 : options.Nt
            if options.complexType > 1
                options.Q{kk} = complex((temp * real(options.Q{kk}) * temp'), (temp * imag(options.Q{kk}) * temp'));
            else
                options.Q{kk} = (temp * options.Q{kk} * temp');
            end
        end
    elseif ~issparse(options.Q)
        %         temp = spdiags([ones(options.N,1) *.5 * options.deltaT^2,ones(options.N,1) * options.deltaT], [0, -options.N], options.N * 2, options.N);
        apu = real(options.Q);
        if options.complexType > 1
            apui = imag(options.Q);
        end
        options.Q = cell(size(options.Q,2),1);
        for kk = 1 : size(options.Q,2)
            if options.complexType > 1
                options.Q{kk} = complex((temp * spdiags(apu(:,kk), 0, options.N, options.N) * temp'), (temp * spdiags(apui(:,kk), 0, options.N, options.N) * temp'));
            else
                options.Q{kk} = (temp * spdiags(apu(:,kk), 0, options.N, options.N) * temp');
            end
        end
        %     options.D = zeros(options.N * 2,size(options.Q,2),'single');
        %     for tt = 1 : size(options.Q,2)
        %         options.D(1 : options.N,tt) = single((.5 * options.deltaT^2) * options.Q(:,tt));
        %         options.D(options.N + 1 : options.N * 2,tt) = single(options.deltaT * options.Q(:,tt));
        %     end
        %     options.Drow = int32(0:options.N*2)';
        %     options.Dcol = zeros(options.N * 2, 1, 'int32');
        %     options.Dcol(1 : options.N) = int32(0 : options.N - 1);
        %     options.Dcol(options.N + 1 : options.N * 2) = int32(options.N + 1 : options.N * 2);
    end
    %     if options.algorithm == 1 || options.algorithm == 2 || options.complexType == 3
    %         options.F = ones(options.N * 3,1,'single');
    %         options.F(2 : 2 : options.N * 2) = single(options.deltaT);
    %         options.Frow = [int32(0:2:options.N*2)';int32(options.N*2 + 1: options.N*3)'];
    %         options.Fcol = ones(options.N * 3,1,'int32');
    %         options.Fcol(1 : 2 : options.N * 2) = int32(0 : options.N - 1);
    %         options.Fcol(2 : 2 : options.N * 2) = int32(options.N : options.N * 2 - 1);
    %         options.Fcol(options.N * 2 + 1 : end) = int32(options.N * 2 : options.N * 3 - 1);
    %     else
    %         options.F = ones(options.N * 2,1,'single');
    %         options.F(2 : 2 : options.N * 2) = single(options.deltaT);
    %         options.Frow = int32(0:2:options.N*2)';
    %         options.Fcol = ones(options.N * 2,1,'int32');
    %         options.Fcol(1 : 2 : options.N * 2) = int32(0 : options.N - 1);
    %         options.Fcol(2 : 2 : options.N * 2) = int32(options.N : options.N * 2 - 1);
    %     end
    %     if options.complexType == 3
    %         options.F = repmat(options.F, 2,1);
    %         options.Frow = [options.Frow;options.Frow(2:end) + max(options.Frow)];
    %         options.Fcol = [options.Fcol;options.Fcol + max(options.Frow) + 1];
    %     end
    if options.algorithm >= 1
        options.F = ones(N3 * 3,1,'single');
    else
        options.F = ones(N3 * 2,1,'single');
    end
    options.F(2 : 2 : N3*2) = single(options.deltaT);
    options.Frow = int32(0:2:N3*2)';
    if options.algorithm >= 1
        options.Frow = [options.Frow;(options.Frow(end) + 1 : int32(N3*3))'];
    end
    options.Fcol = ones(N3 * 2,1,'int32');
    options.Fcol(1 : 2 : end) = int32(0 : N3 - 1);
    options.Fcol(2 : 2 : end) = int32(N3 : N3 * 2 - 1);
    if options.algorithm >= 1
        options.Fcol = [options.Fcol;(int32(N3:N3*2 - 1))'];
    end
end
if options.regularization == 2 || (isfield(options,'F') && ~isempty(options.F))
    options.useF = true;
    if isfield(options,'F') && issparse(options.F)
        options.sparseF = true;
        options.F = options.F.';
    elseif options.useKinematicModel
        options.sparseF = true;
    else
        options.sparseF = false;
    end
    if ~options.useKinematicModel && ~iscell(options.F)
        options.F = {options.F};
    end
    if length(options.F) > 1 && ~options.useKinematicModel
        options.tvF = true;
    else
        options.tvF = false;
    end
    if (iscell(options.F) && isreal(options.F{1})) || options.useKinematicModel
        options.complexF = false;
    else
        options.complexF = true;
    end
else
    options.useF = false;
    options.sparseF = false;
    options.tvF = false;
    options.complexF = false;
end
if isfield(options,'L') && issparse(options.L)
    options.sparseL = true;
    options.L = options.L';
else
    options.sparseL = false;
end
if issparse(options.Q) || (iscell(options.Q) && issparse(options.Q{1}))
    options.sparseQ = true;
    if iscell(options.Q)
        for kk = 1 : length(options.Q)
            options.Q{kk} = options.Q{kk}.';
        end
    else
        options.Q = {options.Q.'};
    end
else
    options.sparseQ = false;
end
if issparse(options.R) || (iscell(options.R) && issparse(options.R{1}))
    options.sparseR = true;
    if iscell(options.R)
        for kk = 1 : length(options.R)
            options.R{kk} = options.R{kk}.';
        end
    else
        options.R = {options.R.'};
    end
else
    options.sparseR = false;
end
if issparse(options.Q) && ~options.tvQ && ~iscell(options.Q)
    options.Q = {options.Q};
end
if options.fadingAlpha == 1 || isempty(options.fadingAlpha)
    options.fadingMemory = false;
else
    options.fadingMemory = true;
end
if isfield(options,'u') && ~isempty(options.u)
    options.useU = true;
    options.u = single(options.u);
    if size(options.u,2) > 1
        options.tvU = true;
    else
        options.tvU = false;
    end
    if isreal(options.u)
        options.complexU = false;
    else
        options.complexU = true;
    end
else
    options.useU = false;
    options.complexU = false;
    options.tvU = false;
end
if isfield(options,'G') && ~isempty(options.G)
    options.useG = true;
    if ~issparse(options.G) && ~iscell(options.G)
        options.G = single(options.G);
    end
    if issparse(options.G) || iscell(options.G) && issparse(options.G{1})
        options.sparseG = true;
        if ~iscell(options.G)
            options.G = options.G';
        else
            for kk = 1 : length(options.G)
                options.G{kk} = options.G{kk}.';
            end
        end
    else
        options.sparseG = false;
    end
    if ~iscell(options.G)
        options.G = {options.G};
    end
    if length(options.G) > 1
        options.tvG = true;
    else
        options.tvG = false;
    end
    if isreal(options.G)
        options.complexG = false;
    else
        options.complexG = true;
        if options.complexType == 3
            if iscell(options.G)
                for kk = 1 : length(options.G)
                    options.G{kk} = [real(options.G{kk}) -imag(options.G{kk}); imag(options.G{kk}) real(options.G{kk})];
                end
            else
                options.G = [real(options.G) -imag(options.G); imag(options.G) real(options.G)];
            end
        end
    end
else
    options.useG = false;
    options.tvG = false;
    options.sparseG = false;
    options.complexG = false;
end
% Invert Q
if (options.algorithm == 1 || options.algorithm == 9 || options.algorithm == 11)
    if ~options.invertedQ
        if issparse(options.Q) || (iscell(options.Q) && issparse(options.Q{1}))
            if iscell(options.Q)
                for kk = 1 : length(options.Q)
                    if options.complexType == 2 || options.complexType == 3
                        options.Q{kk} = complex(inv(real(options.Q{kk})'),inv(imag(options.Q{kk})'));
                    else
                        options.Q{kk} = inv(options.Q{kk});
                    end
                    options.Q{kk}(abs(options.Q{kk}) < 1e-9) = 0;
                    options.Q{kk} = sparse(options.Q{kk});
                    if full(sum(isnan(options.Q{kk}))) > 0
                        error('NaN values detected in the inverted Q covariance, aborting')
                    end
                end
            else
                if options.complexType == 2 || options.complexType == 3
                    options.Q = complex(inv(real(options.Q)),inv(imag(options.Q)));
                else
                    options.Q = inv(options.Q);
                end
                options.Q(abs(options.Q) < 1e-9) = 0;
                options.Q = sparse(options.Q);
                if full(sum(isnan(options.Q))) > 0
                    error('NaN values detected in the inverted Q covariance, aborting')
                end
            end
        else
            if iscell(options.Q)
                for kk = 1 : length(options.Q)
                    if options.complexType == 2 || options.complexType == 3
                        options.Q{kk} = complex(1 ./ real(options.Q{kk}), 1 ./ imag(options.Q{kk}));
                    else
                        options.Q{kk} = 1 ./ options.Q{kk};
                    end
                end
            else
                if options.complexType == 2 || options.complexType == 3
                    options.Q = complex(1 ./ real(options.Q), 1 ./ imag(options.Q));
                else
                    options.Q = 1 ./ options.Q;
                end
            end
        end
    end
    if ~options.invertedR
        if issparse(options.R) || (iscell(options.R) && issparse(options.R{1}))
            if iscell(options.R)
                for kk = 1 : length(options.R)
                    if options.complexType == 2 || options.complexType == 3
                        options.R{kk} = complex(inv(real(options.R{kk})),inv(imag(options.R{kk})));
                    else
                        options.R{kk} = inv(options.R{kk});
                    end
                    options.R{kk}(abs(options.R{kk}) < 1e-9) = 0;
                    options.R{kk} = sparse(options.R{kk});
                    if full(sum(isnan(options.R{kk}))) > 0
                        error('NaN values detected in the inverted R covariance, aborting')
                    end
                end
            else
                if options.complexType == 2 || options.complexType == 3
                    options.R = complex(inv(real(options.R)),inv(imag(options.R)));
                else
                    options.R = inv(options.R);
                end
                options.R(abs(options.R) < 1e-9) = 0;
                options.R = sparse(options.R);
                if full(sum(isnan(options.R))) > 0
                    error('NaN values detected in the inverted R covariance, aborting')
                end
            end
        else
            if iscell(options.R)
                for kk = 1 : length(options.R)
                    if options.complexType == 2 || options.complexType == 3
                        options.R{kk} = complex(1 ./ real(options.R{kk}), 1 ./ imag(options.R{kk}));
                    else
                        options.R{kk} = 1 ./ options.R{kk};
                    end
                end
            else
                if options.complexType == 2 || options.complexType == 3
                    options.R = complex(1 ./ real(options.R), 1 ./ imag(options.R));
                else
                    options.R = 1 ./ options.R;
                end
            end
        end
        options.RR = 1 ./ options.RR;
    end
end
if options.regularization == 1
    if iscell(options.R) && issparse(options.R{1})
        if isreal(options.R{1})
            for kk = 1 : length(options.R)
                [i,j,v] = find(options.R{kk});
                options.R{kk} = sparse([i;(options.Nm + 1:options.N + options.Nm)'], [j;(options.Nm + 1:options.N + options.Nm)'], ...
                    [v;ones(options.N,1)*options.RR], options.N + options.Nm, options.N + options.Nm);
            end
        else
            for kk = 1 : length(options.R)
                [i,j,v] = find((options.R{kk}));
                options.R{kk} = sparse([i;(options.Nm + 1:options.N + options.Nm)'], [j;(options.Nm + 1:options.N + options.Nm)'], ...
                    [v;complex(ones(options.N,1)*options.RR, ones(options.N,1)*options.RR)], options.N + options.Nm, options.N + options.Nm);
            end
        end
        %     else
        %         for kk = 1 : size(options.R,2)
        %             if isreal(options.R)
        %                 options.R(:,kk) = [options.R(:,kk);ones(options.N,1)];
        %             else
        %                 options.R(:,kk) = [options.R(:,kk);complex(ones(options.N,1),ones(options.N,1))];
        %             end
        %         end
    end
end
if isfield(options,'referenceImage') && ~isempty(options.referenceImage)
    options.referenceImage = single(options.referenceImage);
end
if isempty(options.weights) && options.regularization > 2 && (options.prior == 5 || options.prior == 6)
    error('options.weights must be non-empty when using Quadratic or Huber prior')
end
if options.regularization > 2 && (options.prior == 5 || options.prior == 6) && numel(options.weights) ~= ((options.Ndx*2 + 1) * (options.Ndy * 2 + 1) * (options.Ndz * 2 + 1))
    error(['options.weights must have ' num2str((options.Ndx*2 + 1) * (options.Ndy * 2 + 1) * (options.Ndz * 2 + 1)) ' number of elements'])
else
    options.weights = single(options.weights);
end
if isa(options.x0,'double')
    options.x0 = single(options.x0);
end
if isfield(options, 'u') && isa(options.u,'double')
    options.u = single(options.u);
end
if options.useKinematicModel
    if options.complexType > 0
        options.x0 = [options.x0; complex(zeros(options.N, 1, 'single'))];
    else
        options.x0 = [options.x0; zeros(options.N, 1, 'single')];
    end
end
if options.complexType == 3 && isfield(options,'F') && iscell(options.F) && (issparse(options.F{1}) && size(options.F{1},1) == options.N) && ~options.useKinematicModel
    if isreal(options.F{1})
        for kk = 1 : length(options.F)
            [i,j,v] = find(options.F{kk});
            options.F{kk} = sparse([i;i+options.N], [j;j+options.N],[v;v],size(options.F{kk},1)*2, size(options.F{kk},1)*2);
        end
    else
        for kk = 1 : length(options.F)
            [i,j,v] = find(real(options.F{kk}));
            [i2,j2,v2] = find(imag(options.F{kk}));
            if  options.regularization == 2
                options.F{kk} = sparse([i;i2+options.N], [j;j2+options.N],[v;v2],size(options.F{kk},1)*2, size(options.F{kk},1)*2);
            else
                options.F{kk} = sparse([i;i2;i2+options.N;i+options.N], [j;j2+options.N;j2;j+options.N],[v;-v2;v2;v],size(options.F{kk},1)*2, size(options.F{kk},1)*2);
            end
        end
    end
end
if options.complexType == 3 && iscell(options.Q) && (issparse(options.Q{1}) && size(options.Q{1},1) == options.N)
    if isreal(options.Q{1})
        for kk = 1 : length(options.Q)
            [i,j,v] = find(options.Q{kk});
            options.Q{kk} = sparse([i;i+options.N], [j;j+options.N],[v;v],size(options.Q{kk},1)*2, size(options.Q{kk},1)*2);
        end
    else
        for kk = 1 : length(options.Q)
            [i,j,v] = find(real(options.Q{kk}));
            [i2,j2,v2] = find(imag(options.Q{kk}));
            options.Q{kk} = sparse([i;i2+options.N], [j;j2+options.N],[v;v2],size(options.Q{kk},1)*2, size(options.Q{kk},1)*2);
        end
    end
end
if options.complexType == 3 && options.regularization == 2
    if iscell(options.Q)
        for kk = 1 : length(options.Q)
            options.Q{kk} = options.Q{kk} * options.F{kk};
        end
    else
        Q = cell(size(options.Q, 2),1);
        for kk = 1 : size(options.Q, 2)
            apu = double([real(options.Q(:,kk));imag(options.Q(:,kk))]);
            Q{kk} = apu .* options.F{kk};
        end
        options.Q = Q;
        clear Q
    end
end
if options.complexType == 3 && iscell(options.R) && (issparse(options.R{1}) && size(options.R{1},1) == options.Nm)
    if isreal(options.R{1})
        for kk = 1 : length(options.R)
            [i,j,v] = find(options.R{kk});
            options.R{kk} = sparse([i;i+options.N], [j;j+options.N],[v;v],size(options.R{kk},1)*2, size(options.R{kk},1)*2);
        end
    else
        for kk = 1 : length(options.R)
            [i,j,v] = find(real(options.R{kk}));
            [i2,j2,v2] = find(imag(options.R{kk}));
            options.R{kk} = sparse([i;i2+options.N], [j;j2+options.N],[v;v2],size(options.R{kk},1)*2, size(options.R{kk},1)*2);
        end
    end
end
if (isfield(options, 'F') && iscell(options.F) && isreal(options.F{1})) || (isfield(options, 'F') && ~iscell(options.F) && isreal(options.F))
    options.complexF = false;
else
    options.complexF = true;
    %     if ~options.useKinematicModel && options.complexType == 3
    %         if iscell(options.F)
    %             for kk = 1 : length(options.F)
    %                 options.F{kk} = [real(options.F{kk}) -imag(options.F{kk}); imag(options.F{kk}) real(options.F{kk})];
    %             end
    %         else
    %             options.F = [real(options.F) -imag(options.F); imag(options.F) real(options.F)];
    %         end
    %     end
end
% if options.algorithm >= 3 && options.algorithm <= 6
%     if (iscell(options.R) && issparse(options.R{1})) || issparse(options.R)
%         error('Sparse observation noise matrix R is not supported with algorithms 3-6')
%     end
% end
if options.algorithm == 5 || options.algorithm == 6
    if iscell(options.R)
        options.R2 = cell(size(options.R));
        for kk = 1 : length(options.R)
            if options.invertedR
                options.R2{kk} = inv(options.R{kk});
            else
                options.R2{kk} = options.R{kk};
            end
        end
    else
        if options.invertedR
            options.R2 = inv(options.R);
        else
            options.R2 = options.R;
        end
    end
else
    options.R2 = [];
end
if ((options.algorithm >= 7 && options.algorithm <= 8) || options.algorithm == 5) && ~options.invertedR
    if iscell(options.R)
        for kk = 1 : length(options.R)
            if issparse(options.R{kk})
                if options.complexType == 2 || options.complexType == 3
                    options.R{kk} = complex(inv(real(options.R{kk})),inv(imag(options.R{kk})));
                else
                    options.R{kk} = inv(options.R{kk});
                end
            else
                if options.complexType == 2 || options.complexType == 3
                    options.R{kk} = complex(1 ./ real(options.R{kk}), 1 ./ imag(options.R{kk}));
                else
                    options.R{kk} = 1 ./ options.R{kk};
                end
            end
        end
    else
        if issparse(options.R)
            if options.complexType == 2 || options.complexType == 3
                options.R = complex(inv(real(options.R)),inv(imag(options.R)));
            else
                options.R = inv(options.R);
            end
        else
            if options.complexType == 2 || options.complexType == 3
                options.R = complex(1 ./ real(options.R), 1 ./ imag(options.R));
            else
                options.R = 1 ./ options.R;
            end
        end
    end
    options.RR = 1 / options.RR;
end
if options.algorithm == 6 && ~options.invertedR
    if iscell(options.R)
        for kk = 1 : length(options.R)
            if issparse(options.R{kk})
                if options.complexType == 2 || options.complexType == 3
                    options.R{kk} = complex(inv(real(chol(options.R{kk}))),inv(imag(chol(options.R{kk}))));
                else
                    options.R{kk} = inv(chol(options.R{kk}));
                end
            else
                if options.complexType == 2 || options.complexType == 3
                    options.R{kk} = complex(sqrt(1 ./ real(options.R{kk})), sqrt(1 ./ imag(options.R{kk})));
                else
                    options.R{kk} = sqrt(1 ./ options.R{kk});
                end
            end
        end
    else
        if issparse(options.R)
            if options.complexType == 2 || options.complexType == 3
                options.R = complex(inv(real(chol(options.R))),inv(imag(chol(options.R))));
            else
                options.R = inv(chol(options.R));
            end
        else
            if options.complexType == 2 || options.complexType == 3
                options.R = complex(sqrt(1 ./ real(options.R)), sqrt(1 ./ imag(options.R)));
            else
                options.R = sqrt(1 ./ options.R);
            end
        end
    end
    options.RR = 1 / sqrt(options.RR);
end
if options.algorithm == 9 && isvector(options.P0)
    options.P0 = options.P0(1:options.reducedBasisN + 1);
end
if isfield(options,'L') && isreal(options.L)
    options.complexL = false;
else
    options.complexL = true;
end
if isfield(options,'L') && ~options.sparseL && ~options.sparseS
    options.L = single(options.L);
end
if options.complexRef
    options.CI = imag(options.C);
    options.etaI = imag(options.eta);
    options.SATVPhiI = imag(options.SATVPhi);
    options.TVsmoothingI = imag(options.TVsmoothing);
end
if iscell(options.Q)
    if isreal(options.Q{1})
        options.complexQ = false;
    else
        options.complexQ = true;
    end
    if issparse(options.Q{1})
        options.sparseQ = true;
    else
        options.sparseQ = false;
    end
else
    if isreal(options.Q)
        options.complexQ = false;
    else
        options.complexQ = true;
    end
    if issparse(options.Q)
        options.sparseQ = true;
    else
        options.sparseQ = false;
    end
end
if options.algorithm == 12
    apu = options.Q;
    if iscell(apu)
        options.Q = cell(max(options.matCycles,size(options.Q,1)), 1);
    else
        options.Q = cell(max(options.matCycles,size(options.Q,2)), 1);
    end
    ll = 1;
    ii = 1;
    if iscell(apu) && issparse(apu{1})
        for kk = 1 : size(options.Q,1)
            if issparse(options.H)
                if isreal(options.H)
                    options.Q{kk} = (apu{ll} * options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm))';
                else
                    if options.complexType == 2
                        options.Q{kk} = complex((real(apu{ll}) * real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))', (imag(apu{ll}) * imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))');
                    else
                        options.Q{kk} = [(real(apu{ll}) * real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))' (real(apu{ll}) * -imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))';...
                            (imag(apu{ll}) * real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))',(imag(apu{ll}) * imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))'];
                    end
                end
            else
                error('Only sparse system matrices are supported')
%                 if isreal(options.H)
%                     options.Q{kk} = (apu{ll} * options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)')';
%                 else
%                     if options.complexType == 2
%                         options.Q{kk} = complex((real(apu{ll}) * real(options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)'))', (imag(apu{ll}) * imag(options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)'))');
%                     else
%                         options.Q{kk} = [(real(apu{ll}) * real(options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)'))' (real(apu{ll}) * -imag(options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)'))';...
%                             (imag(apu{ll}) * real(options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)'))',(imag(apu{ll}) * imag(options.H(1 + (ii - 1)*options.Nm: ii * options.Nm, :)'))'];
%                     end
%                 end
            end
            ll = ll + 1;
            ii = ii + 1;
            if mod(kk, size(apu,2)) == 0
                ll = 1;
            end
            if mod(kk, options.matCycles) == 0
                ii = 1;
            end
        end
    else
        for kk = 1 : size(options.Q,1)
            if issparse(options.H)
                if isreal(options.H)
                    options.Q{kk} = (double(apu(:,ll)) .* options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm))';
                else
                    if options.complexType == 2
                        options.Q{kk} = complex((real(double(apu(:,ll))) .* real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))', (imag(double(apu(:,ll))) .* imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))');
                    else
                        options.Q{kk} = [(real(double(apu(:,ll))) .* real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))' (real(double(apu(:,ll))) .* -imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))';...
                            (imag(double(apu(:,ll))) .* real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))',(imag(double(apu(:,ll))) .* imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))'];
                    end
                end
            else
                error('Only sparse system matrices are supported')
%                 if isreal(options.H)
%                     options.Q{kk} = (double(apu(:,ll)) .* options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm))';
%                 else
%                     if options.complexType == 2
%                         options.Q{kk} = complex((real(double(apu(:,ll))) .* real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))', (imag(double(apu(:,ll))) .* imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))');
%                     else
%                         options.Q{kk} = [(real(double(apu(:,ll))) .* real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))' (real(double(apu(:,ll))) .* -imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))';...
%                             (imag(double(apu(:,ll))) .* real(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))',(imag(double(apu(:,ll))) .* imag(options.H(:,1 + (ii - 1)*options.Nm: ii * options.Nm)))'];
%                     end
%                 end
            end
            ll = ll + 1;
            ii = ii + 1;
            if mod(kk, size(apu,2)) == 0
                ll = 1;
            end
            if mod(kk, options.matCycles) == 0
                ii = 1;
            end
        end
    end
    options.sparseQ = true;
    if size(options.Q,1) > 1
        options.tvQ = true;
    end
end

% uusi = cell(options.matCycles,1);
% for kk = 1 : options.matCycles
%     uusi{kk} = options.H(:, (kk - 1) * options.Nm + 1 : kk * options.Nm);
% end
% options.H = single(full(uusi{1}))';
if options.backend == 0
    [xt, xs, varargout{1}] = OMEGAKF_OpenCL(options);
elseif options.backend == 1
    [xt, xs, varargout{1}] = OMEGAKF_CUDA(options);
elseif options.backend == 2
    [xt, xs, varargout{1}] = OMEGAKF_CPU(options);
end
if options.complexType == 0
    xt = real(xt);
    xs = real(xs);
end
xt = reshape(xt, options.Nx, options.Ny, options.Nz, [], options.Nt + 2 - options.window);
xt = permute(xt, [1 2 3 5 4]);
xt = squeeze(xt);
if options.useKS
    xs = reshape(xs, options.Nx, options.Ny, options.Nz, options.Nt);
    xs = squeeze(xs);
end
end

