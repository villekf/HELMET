#define MEX_DOUBLE_HANDLE
#include "arrayfire.h"
#include "mex.h"
#include <cstring>


using namespace af;


void mexFunction(int nlhs, mxArray* plhs[],
	int nrhs, const mxArray* prhs[])
{

	const mxArray* options = prhs[0];

	array Ured, Sred, V;
	const size_t nCol = mxGetN(mxGetField(options, 0, "Sigma"));
	const size_t nRow = mxGetM(mxGetField(options, 0, "Sigma"));
	const size_t nEle = mxGetNumberOfElements(mxGetField(options, 0, "Sigma"));
	const size_t nSlice = (nCol * nRow) / nEle;
	const array Sigma = array(nRow, nCol, nSlice, (float*)mxGetData(mxGetField(options, 0, "Sigma")));
	svd(Ured, Sred, V, Sigma);

	const mwSize dim[2] = { static_cast<mwSize>(Ured.dims(0)), static_cast<mwSize>(Ured.dims(1))};
	const mwSize dim1[1] = { static_cast<mwSize>(Sred.dims(0)) };

	plhs[0] = mxCreateNumericArray(2, dim, mxSINGLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericArray(1, dim1, mxSINGLE_CLASS, mxREAL);
	float* out = (float*)mxGetData(plhs[0]);
	float* out1 = (float*)mxGetData(plhs[1]);

	float* apuU = Ured.host<float>();
	float* apuS = Sred.host<float>();

	std::memcpy(out, apuU, (Ured.dims(0) * Ured.dims(1)) * sizeof(float));
	std::memcpy(out1, apuS, Sred.dims(0) * sizeof(float));

	sync();
	af::deviceGC();
	return;
}