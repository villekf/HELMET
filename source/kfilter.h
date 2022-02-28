#pragma once
#define MEX_DOUBLE_HANDLE
#include "arrayfire.h"
#include "mex.h"
#include <cmath>
#include <cstring>
#include <time.h>
#include <algorithm>

#undef MX_HAS_INTERLEAVED_COMPLEX

void kfilter(const mxArray* options, float* out, float* out2, float* outS, float* out2S, float* outP, float* outP2);

// Struct for the TV-prior
typedef struct TVdata_ {
	af::array s1, s2, s3, s4, s5, s6, s7, s8, s9, reference_image, APLSReference;
	bool TV_use_anatomical;
	float tau, TVsmoothing, T, C, eta, APLSsmoothing, SATVPhi = 0.f;
} TVdata;


// Struct for the TV-prior
typedef struct TGVdata_ {
	float lambda1, lambda2, rho, prox;
} TGVdata;