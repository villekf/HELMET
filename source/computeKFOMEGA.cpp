#include "kfilter.h"

using namespace af;


void mexFunction(int nlhs, mxArray* plhs[],
	int nrhs, const mxArray* prhs[])
{
	if (nrhs != 1)
		mexErrMsgTxt("Invalid number of input arguments.  There must be exactly one.");

	if (nlhs != 3)
		mexErrMsgTxt("Invalid number of output arguments.  There must be exactly three.");

	const mxArray* options = prhs[0];

	const uint32_t device = (uint32_t)mxGetScalar(mxGetField(options, 0, "device"));
	setDevice(device);
	const uint64_t window = (uint64_t)mxGetScalar(mxGetField(options, 0, "window"));
	const uint32_t algorithm = (uint32_t)mxGetScalar(mxGetField(options, 0, "algorithm"));
	const bool useSmoother = (bool)mxGetScalar(mxGetField(options, 0, "useKS"));
	const uint32_t Nz = (uint32_t)mxGetScalar(mxGetField(options, 0, "Nz"));
	const bool use3D = (bool)mxGetScalar(mxGetField(options, 0, "use3D"));
	const uint8_t complexType = (uint8_t)mxGetScalar(mxGetField(options, 0, "complexType"));
	const uint8_t storeCovariance = (uint8_t)mxGetScalar(mxGetField(options, 0, "storeCovariance"));
	const bool useKineticModel = (bool)mxGetScalar(mxGetField(options, 0, "useKinematicModel"));
	const uint32_t cgIter = (uint32_t)mxGetScalar(mxGetField(options, 0, "cgIter"));
	mwSize DimZ = 1ULL;
	if (!use3D && Nz > 1)
		DimZ = static_cast<mwSize>(Nz);

	const uint64_t imDim = (uint64_t)mxGetScalar(mxGetField(options, 0, "N"));
	const uint64_t imDimN = (uint64_t)mxGetScalar(mxGetField(options, 0, "NXYZ"));
	uint64_t imDimU = imDim;
	if (useKineticModel)
		imDimU *= 2;
	const uint64_t Nt = (uint64_t)mxGetScalar(mxGetField(options, 0, "Nt"));
	const mwSize dim[3] = { static_cast<mwSize>(imDimU), static_cast<mwSize>(Nt + 1ULL - (window - 1)), DimZ };
	mwSize dimS[3] = { static_cast<mwSize>(1), static_cast<mwSize>(1), static_cast<mwSize>(1) };
	if (useSmoother) {
		dimS[0] = static_cast<mwSize>(imDim);
		dimS[1] = static_cast<mwSize>(Nt - (window - 1));
		dimS[2] = DimZ;
	}
	mwSize dimP[2] = { static_cast<mwSize>(1), static_cast<mwSize>(1)};
	if (storeCovariance) {
		dimP[0] = static_cast<mwSize>(imDimN);
		if (storeCovariance == 1)
			dimP[1] = static_cast<mwSize>(Nt - (window - 1));
		else
			if (algorithm == 10 || algorithm == 11)
				dimP[1] = static_cast<mwSize>(cgIter);
			else
				dimP[1] = static_cast<mwSize>(imDimN);
	}
	mexPrintf("dim[0] = %d\n", dim[0]);
	mexPrintf("dim[1] = %d\n", dim[1]);
	mexPrintf("dim[2] = %d\n", dim[2]);
	mexEvalString("pause(.0001);");

	float* out2 = nullptr, * out2S = nullptr, * outP2 = nullptr;
	if (complexType == 0) {
		plhs[0] = mxCreateNumericArray(3, dim, mxSINGLE_CLASS, mxREAL);
		plhs[1] = mxCreateNumericArray(3, dimS, mxSINGLE_CLASS, mxREAL);
		plhs[2] = mxCreateNumericArray(2, dimP, mxSINGLE_CLASS, mxREAL);
	}
	else {
		plhs[0] = mxCreateNumericArray(3, dim, mxSINGLE_CLASS, mxCOMPLEX);
		plhs[1] = mxCreateNumericArray(3, dimS, mxSINGLE_CLASS, mxCOMPLEX);
		if (complexType == 1)
			plhs[2] = mxCreateNumericArray(2, dimP, mxSINGLE_CLASS, mxREAL);
		else
			plhs[2] = mxCreateNumericArray(2, dimP, mxSINGLE_CLASS, mxCOMPLEX);
		out2 = (float*)mxGetImagData(plhs[0]);
		out2S = (float*)mxGetImagData(plhs[1]);
		if (complexType > 1)
			outP2 = (float*)mxGetImagData(plhs[2]);
	}
	float* out = (float*)mxGetData(plhs[0]);
	float* outS = (float*)mxGetData(plhs[1]);
	float* outP = (float*)mxGetData(plhs[2]);

	try {
		clock_t time = clock();
		kfilter(options, out, out2, outS, out2S, outP, outP2);
		time = clock() - time;
		mexPrintf("Elapsed time is %f seconds.\n", ((float)time) / CLOCKS_PER_SEC);
		mexEvalString("pause(.0001);");
		//	float* out = real(output).host<float>();
		//	std::memcpy(mxGetData(plhs[0]), out, (outSize1) * (outSize2) * sizeof(float));
		//	float* out2 = imag(output).host<float>();
		//	std::memcpy(mxGetImagData(plhs[0]), out2, (outSize1) * (outSize2) * sizeof(float));

		sync();
		af::deviceGC();
	}
	catch (const std::exception& e) {
		af::deviceGC();
		mexErrMsgTxt(e.what());
	}
}