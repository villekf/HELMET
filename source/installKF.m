function installKF(varargin)
% Compiles the necessary mex-files for HELMET
if isempty(varargin) || isempty(varargin{1})
    if ispc
        if exist('OCTAVE_VERSION','builtin') == 5
            proggis1_orig = 'Program Files';
            proggis2_orig = 'Program Files (x86)';
            proggis1 = 'PROGRA~1';
            proggis2 = 'PROGRA~2';
            af_path = getenv('AF_PATH');
            af_path = strrep(af_path, proggis2_orig, proggis2);
            af_path = strrep(af_path, proggis1_orig, proggis1);
        else
            af_path = getenv('AF_PATH');
            if isempty(af_path)
                af_path = 'C:\Program Files\ArrayFire\v3\';
                if exist(af_path,'dir') ~= 7
                    af_path = 'C:\Program Files (x86)\ArrayFire\v3\';
                    if exist(af_path,'dir') ~= 7
                        af_path = 'C:\ArrayFire\v3\';
                    end
                end
            end
        end
    else
        if exist('/opt/arrayfire/','dir') == 7
            af_path = '/opt/arrayfire';
            af_path_include = '/opt/arrayfire/include/';
        elseif exist('/usr/local/include/af/','dir') == 7
            af_path = '/usr/local';
            af_path_include = '/usr/local/include/af';
        elseif exist('/usr/local/arrayfire/','dir') == 7
            af_path = '/usr/local/arrayfire';
            af_path_include = '/usr/local/arrayfire/include/';
        else
            warning('ArrayFire not found. Please specify AF_PATH with installKF(''AF_PATH'').')
            af_path = '';
            af_path_include = '';
        end
    end
else
    af_path = varargin{1};
    af_path_include = [af_path '/include/'];
end

folder = fileparts(which('installKF.m'));
folder = strrep(folder, '\','/');
af_path = strrep(af_path, '\','/');
af_path_include = strrep(af_path_include, '\','/');
% af_path_include = [af_path '/include'];
if exist('OCTAVE_VERSION','builtin') == 0
    if ispc
        compiler = '';
    elseif ismac
        compiler = '';
    elseif isunix
        cc = mex.getCompilerConfigurations('C++','Selected');
        if isempty(cc)
            error('No C++ compiler selected! Use mex -setup C++ to select a C++ compiler')
        end
        if strcmp(cc.Manufacturer, 'GNU')
            gccV = str2double(cc.Version);
            if verLessThan('matlab', '9.4')
                if gccV >= 5 || isnan(gccV)
                    if exist('/usr/bin/g++-4.9','file') == 2
                        compiler = 'GCC=/usr/bin/g++-4.9';
                    elseif exist('/usr/bin/g++-6','file') == 2
                        compiler = 'GCC=/usr/bin/g++-6';
                    elseif exist('/usr/bin/g++-7/','file') == 2
                        compiler = 'GCC=/usr/bin/g++-7';
                    elseif exist('/usr/bin/g++-8/','file') == 2
                        compiler = 'GCC=/usr/bin/g++-8';
                    else
                        compiler = '';
                    end
                else
                    compiler = '';
                end
            elseif verLessThan('matlab', '9.9')
                if gccV >= 7 || isnan(gccV)
                    if exist('/usr/bin/g++-6/','file') == 2
                        compiler = 'GCC=/usr/bin/g++-6';
                    elseif exist('/usr/bin/g++-7/','file') == 2
                        compiler = 'GCC=/usr/bin/g++-7';
                    elseif exist('/usr/bin/g++-8/','file') == 2
                        compiler = 'GCC=/usr/bin/g++-8';
                    elseif exist('/usr/bin/g++-9/','file') == 2
                        compiler = 'GCC=/usr/bin/g++-9';
                    else
                        compiler = '';
                    end
                else
                    compiler = '';
                end
            else
                compiler = '';
            end
        else
            compiler = '';
        end
    end
    
    try
        %%%%%%%%%%%%%%%%%%%%%% CUDA %%%%%%%%%%%%%%%%%%%%%%
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafcuda', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/computeKFOMEGA.cpp'], [folder '/kfilter.cpp'])
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafcuda', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/OMEGASVD.cpp'])
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafcuda', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/ArrayFire_OpenCL_device_info.cpp'])
        if exist('OCTAVE_VERSION','builtin') == 5
            movefile('OMEGASVD.mex', [folder '/OMEGASVD_CUDA.mex'],'f');
            movefile('computeKFOMEGA.mex', [folder '/OMEGAKF_CUDA.mex'],'f');
        else
            if ispc
                movefile([folder '/ArrayFire_OpenCL_device_info.mexw64'], [folder '/ArrayFire_CUDA_device_info.mexw64'],'f');
                movefile([folder '/OMEGASVD.mexw64'], [folder '/OMEGASVD_CUDA.mexw64'],'f');
                movefile([folder '/computeKFOMEGA.mexw64'], [folder '/OMEGAKF_CUDA.mexw64'],'f');
            else
                movefile([folder '/ArrayFire_OpenCL_device_info.mexa64'], [folder '/ArrayFire_CUDA_device_info.mexa64'],'f');
                movefile([folder '/OMEGASVD.mexa64'], [folder '/OMEGASVD_CUDA.mexa64'],'f');
                movefile([folder '/computeKFOMEGA.mexa64'], [folder '/OMEGAKF_CUDA.mexa64'],'f');
            end
        end
        disp('CUDA support enabled')
    catch
        warning('CUDA support not enabled')
    end
    try
        %%%%%%%%%%%%%%%%%%%%%% CPU %%%%%%%%%%%%%%%%%%%%%%
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafcpu', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/computeKFOMEGA.cpp'], [folder '/kfilter.cpp'])
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafcpu', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/OMEGASVD.cpp'])
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafcpu', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/ArrayFire_OpenCL_device_info.cpp'])
        if exist('OCTAVE_VERSION','builtin') == 5
                movefile('ArrayFire_OpenCL_device_info.mex', [folder '/ArrayFire_CPU_device_info.mex'],'f');
                movefile('OMEGASVD.mex', [folder '/OMEGASVD_CPU.mex'],'f');
            movefile('computeKFOMEGA.mex', [folder '/OMEGAKF_CUDA.mex'],'f');
        else
            if ispc
                movefile([folder '/ArrayFire_OpenCL_device_info.mexw64'], [folder '/ArrayFire_CPU_device_info.mexw64'],'f');
                movefile([folder '/OMEGASVD.mexw64'], [folder '/OMEGASVD_CPU.mexw64'],'f');
                movefile([folder '/computeKFOMEGA.mexw64'], [folder '/OMEGAKF_CPU.mexw64'],'f');
            else
                movefile([folder '/ArrayFire_OpenCL_device_info.mexa64'], [folder '/ArrayFire_CPU_device_info.mexa64'],'f');
                movefile([folder '/OMEGASVD.mexa64'], [folder '/OMEGASVD_CPU.mexa64'],'f');
                movefile([folder '/computeKFOMEGA.mexa64'], [folder '/OMEGAKF_CPU.mexa64'],'f');
            end
        end
        disp('CPU support enabled')
    catch
        warning('CPU support not enabled')
    end
    try
        %%%%%%%%%%%%%%%%%%%%%% OpenCL %%%%%%%%%%%%%%%%%%%%%%
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafopencl', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/computeKFOMEGA.cpp'], [folder '/kfilter.cpp'])
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafopencl', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/OMEGASVD.cpp'])
        mex(compiler, '-largeArrayDims', '-outdir', folder, '-lafopencl', ['-L"' af_path '/lib"'], ['-L"' af_path '/lib64"'], ['-I ' folder], ['-I"' af_path_include '"'], ...
            [folder '/ArrayFire_OpenCL_device_info.cpp'])
        if exist('OCTAVE_VERSION','builtin') == 5
            movefile('ArrayFire_OpenCL_device_info.mex', [folder '/ArrayFire_OpenCL_device_info.mex'],'f');
            movefile('OMEGASVD.mex', [folder '/OMEGASVD_OpenCL.mex'],'f');
            movefile('computeKFOMEGA.mex', [folder '/OMEGAKF_CUDA.mex'],'f');
        else
            if ispc
%                 movefile('ArrayFire_OpenCL_device_info.mexw64', [folder '/ArrayFire_OpenCL_device_info.mexw64'],'f');
                movefile([folder '/OMEGASVD.mexw64'], [folder '/OMEGASVD_OpenCL.mexw64'],'f');
                movefile([folder '/computeKFOMEGA.mexw64'], [folder '/OMEGAKF_OpenCL.mexw64'],'f');
            else
%                 movefile('ArrayFire_OpenCL_device_info.mexa64', [folder '/ArrayFire_OpenCL_device_info.mexa64'],'f');
                movefile([folder '/OMEGASVD.mexa64'], [folder '/OMEGASVD_OpenCL.mexa64'],'f');
                movefile([folder '/computeKFOMEGA.mexa64'], [folder '/OMEGAKF_OpenCL.mexa64'],'f');
            end
        end
        disp('OpenCL support enabled')
    catch
        warning('OpenCL support not enabled')
    end
else
    try
        %%%%%%%%%%%%%%%%%%%%%% CUDA %%%%%%%%%%%%%%%%%%%%%%
        mkoctfile('--mex', '-lafcuda', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/computeKFOMEGA.cpp'], [folder '/kfilter.cpp'])
        mkoctfile('--mex', '-lafcuda', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/OMEGASVD.cpp'])
        mkoctfile('--mex', '-lafcuda', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/ArrayFire_OpenCL_device_info.cpp'])
        if exist('OCTAVE_VERSION','builtin') == 5
            movefile(['OMEGASVD.mex'], [folder '/OMEGASVD_CUDA.mex'],'f');
            movefile(['computeKFOMEGA.mex'], [folder '/OMEGAKF_CUDA.mex'],'f');
            movefile(['ArrayFire_OpenCL_device_info.mex'], [folder '/ArrayFire_CUDA_device_info.mex'],'f');
        end
        disp('CUDA support enabled')
    catch
        warning('CUDA support not enabled')
    end
    try
        %%%%%%%%%%%%%%%%%%%%%% CPU %%%%%%%%%%%%%%%%%%%%%%
        mkoctfile('--mex', '-lafcpu', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/computeKFOMEGA.cpp'], [folder '/kfilter.cpp'])
        mkoctfile('--mex', '-lafcpu', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/OMEGASVD.cpp'])
        mkoctfile('--mex', '-lafcpu', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/ArrayFire_OpenCL_device_info.cpp'])
        if exist('OCTAVE_VERSION','builtin') == 5
            movefile(['ArrayFire_OpenCL_device_info.mex'], [folder '/ArrayFire_CPU_device_info.mex'],'f');
            movefile(['OMEGASVD.mex'], [folder '/OMEGASVD_CPU.mex'],'f');
            movefile(['computeKFOMEGA.mex'], [folder '/OMEGAKF_CPU.mex'],'f');
        end
        disp('CPU support enabled')
    catch
        warning('CPU support not enabled')
    end
    try
        %%%%%%%%%%%%%%%%%%%%%% OpenCL %%%%%%%%%%%%%%%%%%%%%%
        mkoctfile('--mex', '-lafopencl', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/computeKFOMEGA.cpp'], [folder '/kfilter.cpp'])
        mkoctfile('--mex', '-lafopencl', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/OMEGASVD.cpp'])
        mkoctfile('--mex', '-lafopencl', ['-L' af_path '/lib'], ['-L' af_path '/lib64'], ['-I ' folder], ['-I' af_path_include], ...
            [folder '/ArrayFire_OpenCL_device_info.cpp'])
        if exist('OCTAVE_VERSION','builtin') == 5
            movefile(['ArrayFire_OpenCL_device_info.mex'], [folder '/ArrayFire_OpenCL_device_info.mex'],'f');
            movefile(['OMEGASVD.mex'], [folder '/OMEGASVD_OpenCL.mex'],'f');
            movefile(['computeKFOMEGA.mex'], [folder '/OMEGAKF_OpenCL.mex'],'f');
        end
        disp('OpenCL support enabled')
    catch
        warning('OpenCL support not enabled')
    end
end