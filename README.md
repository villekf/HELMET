# HELMET

High-dimensional Kalman filter toolbox (HELMET) is an open-source software for high-dimensional linear dynamic problems. HELMET includes various KF algorithms that can be used to estimate any linear dynamic problems. The core idea of HELMET is the use of GPUs for the fast computation of Kalman filter.

# Introduction

HELMET is a software for [MATLAB](https://www.mathworks.com/) and [GNU Octave](https://www.gnu.org/software/octave/) to estimate especially high-dimensional dynamic problems. However, the software does not have constraints on the input data size or dimensionality, the only requirement is a linear problem. As such problems of any dimension can be used though estimating only a few states gives no benefit when using GPUs. Note also that an inidivual state/estimate can be 1D, 2D or 3D and can also be complex-valued. HELMET heavily uses ArrayFire for the computations and as such can be used with both GPUs and CPUs. For GPUs, both CUDA and OpenCL support is available. The desired "backend" (e.g. CUDA, OpenCL or CPU) can be chosen in the main m-file before running the code.

For "high dimension" the number of estimates (states) is considered to be > 10000 though this depends on the computational resources. Note that the main restrictions in using the regular KF and its modifications is the memory required by the estimate error covariance(s) and Kalman gain. For the error covariance, the memory requirements can be computed with `total number of states * total number of states * 4 / 1024^3`, for the gain the formula is `total number of states * total number of measurements * 4 / 1024^3`.

# Getting Started

See the [documentation](https://github.com/villekf/HELMET/blob/main/documentation.pdf) for help.

# System Requirements

ArrayFire is required (https://arrayfire.com/download/).

MATLAB R2009a or later is mandatory.

For Windows Visual Studio 2022, 2019, 2017 is recommended with "Desktop development with C++", no other options are required. https://visualstudio.microsoft.com/

For Linux it is recommended to use GCC which usually comes bundled with the system. 

On MacOS Xcode is required https://apps.apple.com/us/app/xcode/id497799835?mt=12.

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

# Known Issues and Limitations

Due to ArrayFire bugs, OpenCL might not work with algorithms requiring inversion of matrices or SVD.

# Reporting Bugs and Feature Requests

For any bug reports I recommend posting an issue on GitHub. For proper analysis I need the main-file that you have used and if you have used GATE data then also the macros. Preferably also all possible .mat files created, especially if the problem occurs in the reconstruction phase.

For feature requests, post an issue on GitHub. I do not guarantee that a specific feature will be added in the future.


# Citations

If you wish to use this software in your work, for the moment cite this page. In the future, a proper publication is planned.


# Acknowledgments

Almost all code by Ville-Veikko Wettenhovi. Other relevant sources have been cited in the code. Portions from the [OMEGA](https://github.com/villekf/OMEGA) software are also used in this toolbox. 

This work has been supported by the [University of Eastern Finland](https://www.uef.fi/en) and Academy of Finland.
