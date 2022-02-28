# HELMET

High-dimensional Kalman filter toolbox (HELMET) is an open-source software for high-dimensional linear dynamic problems. HELMET includes various KF algorithms that can be used to estimate any linear dynamic problems.

# Introduction

HELMET is a software for [MATLAB](https://www.mathworks.com/) and [GNU Octave](https://www.gnu.org/software/octave/) to estimate especially high-dimensional dynamic problems. However, the software does not have constraints on the input data size or dimensionality, the only requirement is a linear problem. As such problems of any dimension can be used though estimating only a few states gives no benefit when using GPUs. Note also that the an inidivual state/estimate can be 1D, 2D or 3D.

For "high dimension" the number of estimates (states) is considered to be > 10000 though this depends on the computational resources. Note that the main restrictions in using the regular KF and its modifications is the memory required by the estimate error covariance(s) and Kalman gain. For the error covariance, the memory requirements can be computed with `total number of states * total number of states * 4 / 1024^3`, for the gain the formula is `total number of states * total number of measurements * 4 / 1024^3`.

# Getting Started

See the [documentation](https://github.com/villekf/HELMET/blob/main/documentation.pdf) for help.

# Installation

This software uses ArrayFire library for all the KF computations. You can find AF binaries from here:  
https://arrayfire.com/download/
and the source code from here:  
https://github.com/arrayfire/arrayfire

On Windows you might need to install [Visual Studio 2015 (x64) runtime libraries](https://www.microsoft.com/en-in/download/details.aspx?id=48145) first before installing ArrayFire.

After installing/building ArrayFire, a C++ compiler is needed in order to compile the MEX-files and use this software. Visual Studio and GCC have been tested to work and are recommended depending on your platform (Visual Studio on Windows, GCC on Linux, clang should work on MacOS). Specifically, Visual Studio 2019 have been tested to work on Windows 10 and as well as G++ 7.3 and 9.3 on Ubuntu 20.04. The use of MinGW++ on Windows requires manual compilation of ArrayFire on Windows with MinGW. For instructions on how to do this, see [here](https://github.com/villekf/OMEGA/wiki/Building-ArrayFire-with-Mingw-on-Windows)). Note that Octave support has not yet been implemented.

Visual studio can be downloaded from [here](https://visualstudio.microsoft.com/).

On Ubuntu you can install g++ with `sudo apt install build-essential`.

To install the HELMET software, either simply extract the release/master package, obtain the source code through git: `git clone https://github.com/villekf/HELMET`
and then add the HELMET folder and source subfolder to MATLAB/Octave path. Finally, run `installKF` to build the necessary MEX-files. 

**ArrayFire library paths needs to be on system path when running the mex-files or otherwise the required libraries will not be found.**

For OpenCL, you need drivers/OpenCL runtimes for your device(s). If you have GPUs/APUs then simply having the vendor drivers should be enough. For Intel CPUs without an integrated GPU you need CPU runtimes (see the link below). 

For AMD CPUs it seems that the AMD drivers released around the summer 2018 and after no longer support CPUs so you need an older driver in order to get CPU support or use an alternative runtime. One possibility is to use POCL http://portablecl.org/ and another is to try the Intel runtimes (link below).

Intel runtimes can be found here:
https://software.intel.com/en-us/articles/opencl-drivers

Installing/building ArrayFire to the default location (`C:\Program Files\ArrayFire` on Windows, `/opt/arrayfire/` on Linux/MacOS) should cause `installKF` to automatically locate everything. However, in both cases you need to add the library paths to the system PATH. On Windows you will be prompted for this during the installation, for Linux you need to add `/opt/arrayfire/lib` (bulding from source) or `/opt/arrayfire/lib64` (installer) to the library path (e.g. `sudo ldconfig /opt/arrayfire/lib64/`). Alternatively, on Linux, you can also build/install it directly into the `/usr/local/` folder (requires sudo rights) thus avoiding the need to add to the system path.
