#include "kfilter.h"

using namespace af;

// Various batch functions
array batchMul(const array& lhs, const array& rhs) {
	return lhs * rhs;
}

array batchDiv(const array& lhs, const array& rhs) {
	return lhs / rhs;
}

array batchMinus(const array& lhs, const array& rhs) {
	return lhs - rhs;
}

array batchPlus(const array& lhs, const array& rhs) {
	return lhs + rhs;
}

array matmul3(const std::vector<array>& S, const std::vector<array>& Si, const array& A, const uint64_t ind, const bool trans = false) {
	array output;
	if (trans)
		output = join(0, matmul(S[ind], A(seq(0, A.dims(0) / 2 - 1), span), AF_MAT_TRANS) + matmul(Si[ind], A(seq(A.dims(0) / 2, end), span), AF_MAT_TRANS),
			matmul(S[ind], A(seq(A.dims(0) / 2, end), span), AF_MAT_TRANS) - matmul(Si[ind], A(seq(0, A.dims(0) / 2 - 1), span), AF_MAT_TRANS));
	else
		output = join(0, matmul(S[ind], A(seq(0, A.dims(0) / 2 - 1), span)) - matmul(Si[ind], A(seq(A.dims(0) / 2, end), span)),
			matmul(Si[ind], A(seq(0, A.dims(0) / 2 - 1), span)) + matmul(S[ind], A(seq(A.dims(0) / 2, end), span)));
	return output;
}

array matmul39(const std::vector<array>& S, const std::vector<array>& Si, const array& A, const uint64_t ind, const bool trans = false) {
	array output;
	if (trans)
		output = join(0, matmul(S[ind], A, AF_MAT_TRANS) + matmul(Si[ind], A, AF_MAT_TRANS),
			matmul(S[ind], A, AF_MAT_TRANS) - matmul(Si[ind], A, AF_MAT_TRANS));
	else
		output = join(0, join(1, matmul(S[ind], A), -matmul(Si[ind], A)),
			join(1, matmul(Si[ind], A), matmul(S[ind], A)));
	return output;
}

array matmul392(const array Pred, const array& A, const uint8_t complexType = 0, const bool trans = false) {
	array output;
	//if (trans)
		//output = join(0, matmul(S[ind], A, AF_MAT_TRANS) + matmul(Si[ind], A, AF_MAT_TRANS),
		//	matmul(S[ind], A, AF_MAT_TRANS) - matmul(Si[ind], A, AF_MAT_TRANS));
	//else
	if (complexType == 3)
		output = join(0, matmul(Pred, A(seq(0, A.dims(0) / 2 - 1), span)), matmul(Pred, A(seq(A.dims(0) / 2, end), span)));
	else
		output = matmul(Pred, A);
	return output;
}

array vecmul3(const std::vector<array>& S, const std::vector<array>& Si, const array& A, const uint64_t ind, const bool trans = false) {
	array output;
	if (trans)
		output = join(0, matmul(S[ind], A(seq(0, A.dims(0) / 2 - 1)), AF_MAT_TRANS) + matmul(Si[ind], A(seq(A.dims(0) / 2, end)), AF_MAT_TRANS),
			matmul(S[ind], A(seq(A.dims(0) / 2, end)), AF_MAT_TRANS) - matmul(Si[ind], A(seq(0, A.dims(0) / 2 - 1)), AF_MAT_TRANS));
	else
		output = join(0, matmul(S[ind], A(seq(0, A.dims(0) / 2 - 1))) - matmul(Si[ind], A(seq(A.dims(0) / 2, end))),
			matmul(S[ind], A(seq(A.dims(0) / 2, end))) + matmul(Si[ind], A(seq(0, A.dims(0) / 2 - 1))));
	return output;
}

double* loadDoubles(const mxArray* options, const char* var) {
#ifdef MX_HAS_INTERLEAVED_COMPLEX
	return (double*)mxGetDoubles(mxGetField(options, 0, var));
#else
	return (double*)mxGetData(mxGetField(options, 0, var));
#endif
}

double* loadComplexDoubles(const mxArray* options, const char* var) {
#ifdef MX_HAS_INTERLEAVED_COMPLEX
	return (double*)mxGetComplexDoubles(mxGetField(options, 0, var));
#else
	return (double*)mxGetImagData(mxGetField(options, 0, var));
#endif
}

float* loadFloats(const mxArray* options, const char* var) {
#ifdef MX_HAS_INTERLEAVED_COMPLEX
	return (float*)mxGetSingles(mxGetField(options, 0, var));
#else
	return (float*)mxGetData(mxGetField(options, 0, var));
#endif
}

float* loadComplexFloats(const mxArray* options, const char* var, float* imagData = nullptr) {
#ifdef MX_HAS_INTERLEAVED_COMPLEX
	mxComplexSingle* input = mxGetComplexSingles(mxGetField(options, 0, var));
#else
	return (float*)mxGetImagData(mxGetField(options, 0, var));
#endif
}

// Modified Gram-Schmidt orthogonalization
array MGSOG(const array X) {
	const size_t d = X.dims(0);
	const size_t n = X.dims(1);
	const size_t m = std::min(d, n);
	//array R = identity(m, n);
	array Q = constant(0.f, d, m);
	array D = constant(0.f, 1, m);
	for (int64_t i = 0; i < m; i++) {
		array v = X(span, i);
		for (int64_t j = 0; j < i - 1; j++) {
			//R(j, i) = matmulTN(Q(span, j), v) / D(j);
			//v -= tile(R(j, i), Q.dims(0), 1) * Q(span, j);
			v -= tile(matmulTN(Q(span, j), v) / D(j), Q.dims(0), 1) * Q(span, j);
		}
		Q(span, i) = v;
		D(i) = dot(Q(span, i), Q(span, i));
	}
	return Q;
}

// Create a complex array
array kompleksiset(const size_t numRows, const size_t numCols, const float* HH1, const float* HH2, const size_t numSlices = 1ULL) {
	array H2(numRows, numCols, numSlices, HH1, afHost);
	array H3(numRows, numCols, numSlices, HH2, afHost);

	return complex(H2, H3);
}

// Padding
af::array padding(const af::array& im, const uint32_t Nx, const uint32_t Ny, const uint32_t Nz, const uint32_t Ndx, const uint32_t Ndy, const uint32_t Ndz,
	const bool zero_pad = false)
{
	af::array padd;
	if (zero_pad == 1) {
		af::dtype type = im.type();
		if (Nz == 1) {
			if (im.dims(1) == 1)
				padd = moddims(im, Nx, Ny, Nz);
			else
				padd = im;
			padd = moddims(im, Nx, Ny, Nz);
			af::array out = af::constant(0, padd.dims(0) + 2 * Ndx, padd.dims(1) + 2 * Ndy, type);
			out(static_cast<double>(Ndx) + af::seq(static_cast<double>(padd.dims(0))), static_cast<double>(Ndy) + af::seq(static_cast<double>(padd.dims(1)))) = padd;
			padd = out;
		}
		else {
			if (im.dims(2) == 1)
				padd = moddims(im, Nx, Ny, Nz);
			else
				padd = im;
			padd = moddims(im, Nx, Ny, Nz);
			af::array out = af::constant(0, padd.dims(0) + 2 * Ndx, padd.dims(1) + 2 * Ndy, padd.dims(2) + 2 * Ndz, type);
			out(static_cast<double>(Ndx) + af::seq(static_cast<double>(padd.dims(0))), static_cast<double>(Ndy) + af::seq(static_cast<double>(padd.dims(1))),
				static_cast<double>(Ndz) + af::seq(static_cast<double>(padd.dims(2)))) = padd;
			padd = out;
		}
	}
	else {
		if (im.dims(1) == 1)
			padd = moddims(im, Nx, Ny, Nz);
		else
			padd = im;
		af::array out = padd;
		if (Ndx > 0)
			out = af::join(0, af::flip(padd(af::seq(static_cast<double>(Ndx)), af::span, af::span), 0), padd, af::flip(padd(af::seq(static_cast<double>(padd.dims(0) - Ndx), static_cast<double>(padd.dims(0) - 1)), af::span, af::span), 0));
		if (Ndy > 0)
			out = af::join(1, af::flip(out(af::span, af::seq(static_cast<double>(Ndy)), af::span), 1), out, af::flip(out(af::span, af::seq(static_cast<double>(out.dims(1) - Ndy), static_cast<double>(out.dims(1) - 1)), af::span), 1));
		if (Nz == 1 || Ndz == 0) {
		}
		else {
			out = af::join(2, af::flip(out(af::span, af::span, af::seq(static_cast<double>(Ndz))), 2), out, af::flip(out(af::span, af::span, af::seq(static_cast<double>(out.dims(2) - Ndz), static_cast<double>(out.dims(2) - 1))), 2));
		}
		padd = out;
	}
	return padd;
}

// TGV
af::array TGVDenoising(const uint32_t Nx, const uint32_t Ny, const uint32_t Nz, const float lambda1, const float lambda2, const float rho, const float prox, const array& im, const uint32_t nIter) {
	array x2 = moddims(im, Nx, Ny, Nz);
	const float sigma = 1.f / prox / 72.f;
	array x;
	if (Nz <= 1U) {
		array r2 = constant(0.f, Nx, Ny, 2);
		array u2 = constant(0.f, Nx, Ny, 4);

		for (int n = 0; n < nIter; n++) {
			array temp = prox * join(2, join(0, -diff1(u2(span, span, 0)), u2(end, span, 0)) - join(1, u2(span, 0, 1), diff1(u2(span, span, 1), 1)),
				join(1, -diff1(u2(span, span, 2), 1), u2(span, end, 2)) - join(0, u2(0, span, 3), diff1(u2(span, span, 3))));
			x = x2 - (join(0, -1.f * temp(0, span, 0), -diff1(temp(seq(0, end - 1), span, 0)), temp(end - 1, span, 0)) +
				join(1, -1.f * temp(span, 0, 1), -diff1(temp(span, seq(0, end - 1), 1), 1), temp(span, end - 1, 1)));
			x = (x + prox * moddims(im, Nx, Ny, Nz)) / (1.f + prox);
			array r = r2 + temp;
			r = r - batchFunc(r, max(sqrt(sum(r * r, 2)) / (prox * lambda1), constant(1.f, Nx, Ny, Nz)), batchDiv);
			array apux = 2.f * x - x2;
			array u = join(2, join(0, diff1(apux), constant(0.f, 1, Ny)), join(1, diff1(apux, 1), constant(0.f, Nx, 1))) - (2.f * r - r2);
			u = u2 + sigma * join(2, join(0, u(0, span, 0), diff1(u(span, span, 0))), join(1, diff1(u(span, span, 0), 1), constant(0.f, Nx, 1)), join(1, u(span, 0, 1), diff1(u(span, span, 1), 1)),
				join(0, diff1(u(span, span, 1)), constant(0.f, 1, Ny)));
			u = batchFunc(u, max(sqrt(sum(u * u, 2)) / lambda2, constant(0.f, Nx, Ny, Nz)), batchDiv);
			x2 += rho * (x - x2);
			r2 += rho * (r - r2);
			u2 += rho * (u - u2);
		}
	}
	else {
		array r2 = constant(0.f, Nx, Ny, Nz, 3);
		array u2 = constant(0.f, Nx, Ny, Nz, 9);

		for (int n = 0; n < nIter; n++) {
			array temp = prox * join(3, join(0, -diff1(u2(span, span, span, 0)), u2(end, span, span, 0)) - join(1, u2(span, 0, span, 1), diff1(u2(span, span, span, 1), 1)) -
				join(2, u2(span, span, 0, 2), diff1(u2(span, span, span, 2), 2)),
				join(1, -diff1(u2(span, span, span, 3), 1), u2(span, end, span, 3)) - join(0, u2(0, span, span, 4), diff1(u2(span, span, span, 4))) -
				join(2, u2(span, span, 0, 5), diff1(u2(span, span, span, 5), 2)),
				join(2, -diff1(u2(span, span, span, 6), 2), u2(span, span, end, 6)) - join(0, u2(0, span, span, 7), diff1(u2(span, span, span, 7))) -
				join(1, u2(span, 0, span, 8), diff1(u2(span, span, span, 8), 1)));
			//if (n == 0) {
			//	mexPrintf("temp.dims(0) = %d\n", temp.dims(0));
			//	mexPrintf("temp.dims(1) = %d\n", temp.dims(1));
			//	mexPrintf("temp.dims(2) = %d\n", temp.dims(2));
			//	mexEvalString("pause(.0001);");
			//}
			x = x2 - (join(0, -1.f * temp(0, span, span, 0), -diff1(temp(seq(0, end - 1), span, span, 0)), temp(end - 1, span, span, 0)) +
				join(1, -1.f * temp(span, 0, span, 1), -diff1(temp(span, seq(0, end - 1), span, 1), 1), temp(span, end - 1, span, 1)) +
				join(2, -1.f * temp(span, span, 0, 2), -diff1(temp(span, span, seq(0, end - 1), 2), 2), temp(span, span, end - 1, 2)));
			x = (x + prox * moddims(im, Nx, Ny, Nz)) / (1.f + prox);
			//if (n == 0) {
			//	mexPrintf("x.dims(0) = %d\n", x.dims(0));
			//	mexPrintf("x.dims(1) = %d\n", x.dims(1));
			//	mexPrintf("x.dims(2) = %d\n", x.dims(2));
			//	mexEvalString("pause(.0001);");
			//}
			array r = r2 + temp;
			r = r - batchFunc(r, max(sqrt(sum(r * r, 2)) / (prox * lambda1), constant(1.f, Nx, Ny, Nz)), batchDiv);
			//if (n == 0) {
			//	mexPrintf("r.dims(0) = %d\n", r.dims(0));
			//	mexPrintf("r.dims(1) = %d\n", r.dims(1));
			//	mexPrintf("r.dims(2) = %d\n", r.dims(2));
			//	mexEvalString("pause(.0001);");
			//}
			array apux = 2.f * x - x2;
			array u = join(3, join(0, diff1(apux), constant(0.f, 1, Ny, 1)), join(1, diff1(apux, 1), constant(0.f, Nx, 1, 1)), join(2, diff1(apux, 2), constant(0.f, 1, 1, Nz))) - (2.f * r - r2);
			u = u2 + sigma * join(3, join(3, join(0, u(0, span, span, 0), diff1(u(span, span, span, 0))), join(1, diff1(u(span, span, span, 0), 1), constant(0.f, Nx, 1, 1)), join(2, diff1(u(span, span, span, 0), 2), constant(0.f, 1, 1, Nz))),
				join(3, join(1, u(span, 0, span, 1), diff1(u(span, span, span, 1), 1)), join(0, diff1(u(span, span, span, 1)), constant(0.f, Nx, 1, 1)), join(2, diff1(u(span, span, span, 1), 2), constant(0.f, 1, 1, Nz))),
				join(3, join(2, u(span, span, 0, 2), diff1(u(span, span, span, 2), 2)), join(0, diff1(u(span, span, span, 2)), constant(0.f, Nx, 1, 1)), join(1, diff1(u(span, span, span, 2), 1), constant(0.f, 1, Ny, 1))));
			u = batchFunc(u, max(sqrt(sum(u * u, 2)) / lambda2, constant(0.f, Nx, Ny, Nz)), batchDiv);
			//if (n == 0) {
			//	mexPrintf("u.dims(0) = %d\n", u.dims(0));
			//	mexPrintf("u.dims(1) = %d\n", u.dims(1));
			//	mexPrintf("u.dims(2) = %d\n", u.dims(2));
			//	mexEvalString("pause(.0001);");
			//}
			x2 += rho * (x - x2);
			r2 += rho * (r - r2);
			u2 += rho * (u - u2);
		}
	}
	return x;
}

// Compute the TV prior
af::array TVprior(const uint32_t Nx, const uint32_t Ny, const uint32_t Nz, const TVdata& TV, const af::array& ima, const uint32_t TVtype, const uint32_t NN) {
	af::array gradi;

	const af::array im = af::moddims(ima, Nx, Ny, Nz);
	const uint64_t Dim = static_cast<uint64_t>(Nx) * static_cast<uint64_t>(Ny) * static_cast<uint64_t>(Nz);
	// 1st order differentials
	af::array g = af::constant(0.f, Nx, Ny, Nz);
	af::array f = af::constant(0.f, Nx, Ny, Nz);
	af::array h = af::constant(0.f, Nx, Ny, Nz);
	f(af::seq(0, Nx - 2u), af::span, af::span) = -af::diff1(im);
	f(af::end, af::span, af::span) = f(Nx - 2u, af::span, af::span) * -1.f;
	if (Ny > 1) {
		g(af::span, af::seq(0, Ny - 2u), af::span) = -af::diff1(im, 1);
		g(af::span, af::end, af::span) = g(af::span, Ny - 2u, af::span) * -1.f;
	}
	if (Nz > 1) {
		h(af::span, af::span, af::seq(0, Nz - 2u)) = -af::diff1(im, 2);
		h(af::span, af::span, af::end) = h(af::span, af::span, Nz - 2u) * -1.f;
	}

	af::array pval, apu1, apu2, apu3, apu4;
	g = af::flat(g);
	f = af::flat(f);
	h = af::flat(h);

	// If anatomical prior is used
	if ((TV.TV_use_anatomical && TVtype != 4U) || TVtype == 5U) {
		if (TVtype == 1U) {
			if (Nz > 1) {
				pval = (TV.s1 * af::pow(f, 2.) + TV.s5 * af::pow(g, 2.) + TV.s9 * af::pow(h, 2.) + TV.s4 * f * g + TV.s7 * f * h + TV.s2 * f * g + TV.s8 * h * g + TV.s3 * f * h + TV.s6 * h * g + TV.TVsmoothing);
				pval(pval <= 0) = TV.TVsmoothing;
				pval = af::sqrt(pval);
				apu1 = 0.5f * (2.f * TV.s1 * f + TV.s4 * g + TV.s7 * h + TV.s2 * g + TV.s3 * h) / pval;
				apu2 = 0.5f * (2.f * TV.s5 * g + TV.s4 * f + TV.s2 * f + TV.s8 * h + TV.s6 * h) / pval;
				apu3 = 0.5f * (2.f * TV.s9 * h + TV.s8 * g + TV.s6 * g + TV.s7 * f + TV.s3 * f) / pval;
				apu4 = 0.5f * (2.f * TV.s1 * f + 2.f * TV.s5 * g + 2.f * TV.s9 * h + TV.s4 * f + TV.s2 * f + TV.s8 * h + TV.s6 * h + TV.s4 * g + TV.s7 * h + TV.s2 * g + TV.s3 * h
					+ TV.s8 * g + TV.s6 * g + TV.s7 * f + TV.s3 * f) / pval;
			}
			else {
				pval = (TV.s1(seq(Dim * NN, Dim * (NN + 1) - 1)) * af::pow2(f) + TV.s5(seq(Dim * NN, Dim * (NN + 1) - 1)) * af::pow2(g) + TV.s4(seq(Dim * NN, Dim * (NN + 1) - 1)) * f * g + TV.s2(seq(Dim * NN, Dim * (NN + 1) - 1)) * f * g + TV.TVsmoothing);
				pval(pval <= 0) = TV.TVsmoothing;
				//mexPrintf("pval.summa = %f\n", af::sum<float>(flat(pval)));
				////mexPrintf("f.summa = %f\n", af::sum<float>(flat(f)));
				////mexPrintf("g.summa = %f\n", af::sum<float>(flat(g)));
				//mexPrintf("pval.min = %f\n", af::min<float>(flat(pval)));
				//mexPrintf("pval.max = %f\n", af::max<float>(flat(pval)));
				//mexPrintf("NN = %u\n", NN);
				//mexEvalString("pause(.0001);");
				pval = sqrt(pval);
				apu1 = 0.5f * (2.f * TV.s1(seq(Dim * NN, Dim * (NN + 1) - 1)) * f + TV.s4(seq(Dim * NN, Dim * (NN + 1) - 1)) * g + TV.s2(seq(Dim * NN, Dim * (NN + 1) - 1)) * g) / pval;
				apu2 = 0.5f * (2.f * TV.s5(seq(Dim * NN, Dim * (NN + 1) - 1)) * g + TV.s4(seq(Dim * NN, Dim * (NN + 1) - 1)) * f + TV.s2(seq(Dim * NN, Dim * (NN + 1) - 1)) * f) / pval;
				apu4 = 0.5f * (2.f * TV.s1(seq(Dim * NN, Dim * (NN + 1) - 1)) * f + 2.f * TV.s5(seq(Dim * NN, Dim * (NN + 1) - 1)) * g + TV.s4(seq(Dim * NN, Dim * (NN + 1) - 1)) * f + TV.s2(seq(Dim * NN, Dim * (NN + 1) - 1)) * f + TV.s4(seq(Dim * NN, Dim * (NN + 1) - 1)) * g + TV.s2(seq(Dim * NN, Dim * (NN + 1) - 1)) * g) / pval;
				//mexPrintf("apu1.summa = %f\n", af::sum<float>(flat(apu1)));
				//mexPrintf("apu2.summa = %f\n", af::sum<float>(flat(apu2)));
				//mexPrintf("apu4.summa = %f\n", af::sum<float>(flat(apu4)));
				//mexEvalString("pause(.0001);");
			}
		}
		else if (TVtype == 2U) {
			const af::array reference_image = af::moddims(TV.reference_image(seq(Dim * NN, Dim * (NN + 1) - 1)), Nx, Ny, Nz);
			af::array gp = af::constant(0.f, Nx, Ny, Nz);
			af::array fp = af::constant(0.f, Nx, Ny, Nz);
			af::array hp = af::constant(0.f, Nx, Ny, Nz);
			fp(af::seq(0, Nx - 2u), af::span, af::span) = -af::diff1(reference_image);
			fp(af::end, af::span, af::span) = fp(Nx - 2u, af::span, af::span) * -1.f;
			gp(af::span, af::seq(0, Ny - 2u), af::span) = -af::diff1(reference_image, 1);
			gp(af::span, af::end, af::span) = gp(af::span, Ny - 2u, af::span) * -1.f;
			if (Nz > 1) {
				hp(af::span, af::span, af::seq(0, Nz - 2u)) = -af::diff1(reference_image, 2);
				hp(af::span, af::span, af::end) = hp(af::span, af::span, Nz - 2u) * -1.f;
			}

			gp = af::flat(gp);
			fp = af::flat(fp);
			hp = af::flat(hp);

			if (Nz > 1) {
				pval = af::sqrt(af::pow(f, 2.) + af::pow(g, 2.) + af::pow(h, 2.) + TV.T * (af::pow(fp, 2.) + af::pow(gp, 2.) + af::pow(hp, 2.)) + TV.TVsmoothing);
				apu1 = f / pval;
				apu2 = g / pval;
				apu3 = h / pval;
				apu4 = (f + g + h) / pval;
			}
			else {
				pval = af::sqrt(af::pow(f, 2.) + af::pow(g, 2.) + TV.T * (af::pow(fp, 2.) + af::pow(gp, 2.)) + TV.TVsmoothing);
				apu1 = f / pval;
				apu2 = g / pval;
				apu4 = (f + g) / pval;
			}
		}
		// For APLS
		else if (TVtype == 3U) {
			const af::array reference_image = af::moddims(TV.reference_image(seq(Dim * NN, Dim * (NN + 1) - 1)), Nx, Ny, Nz);
			af::array gp = af::constant(0.f, Nx, Ny, Nz);
			af::array fp = af::constant(0.f, Nx, Ny, Nz);
			af::array hp = af::constant(0.f, Nx, Ny, Nz);
			fp(af::seq(0, Nx - 2u), af::span, af::span) = -af::diff1(reference_image);
			fp(af::end, af::span, af::span) = fp(Nx - 2u, af::span, af::span) * -1.f;
			gp(af::span, af::seq(0, Ny - 2u), af::span) = -af::diff1(reference_image, 1);
			gp(af::span, af::end, af::span) = gp(af::span, Ny - 2u, af::span) * -1.f;
			if (Nz > 1) {
				hp(af::span, af::span, af::seq(0, Nz - 2u)) = -af::diff1(reference_image, 2);
				hp(af::span, af::span, af::end) = hp(af::span, af::span, Nz - 2u) * -1.f;
			}

			fp = af::flat(fp);
			gp = af::flat(gp);
			hp = af::flat(hp);

			if (Nz > 1) {
				const af::array epsilon = af::batchFunc(af::join(1, fp, gp, hp), af::sqrt(fp * fp + gp * gp + hp * hp + TV.eta * TV.eta), batchDiv);
				const af::array apu = af::sum(af::join(1, f, g, h) * epsilon, 1);

				pval = (f * f + g * g + h * h - apu * apu + TV.APLSsmoothing);
				pval(pval <= 0.f) = TV.APLSsmoothing;
				pval = af::sqrt(pval);
				apu1 = (f - (apu * epsilon(af::span, 0))) / pval;
				apu2 = (g - (apu * epsilon(af::span, 1))) / pval;
				apu3 = (h - (apu * epsilon(af::span, 2))) / pval;
				apu4 = (f - (apu * epsilon(af::span, 0)) + g - (apu * epsilon(af::span, 1)) + h - (apu * epsilon(af::span, 2))) / pval;
			}
			else {
				const af::array epsilon = af::batchFunc(af::join(1, fp, gp), af::sqrt(fp * fp + gp * gp + TV.eta * TV.eta), batchDiv);
				const af::array apu = af::sum(af::join(1, f, g) * epsilon, 1);

				pval = (f * f + g * g - apu * apu + TV.APLSsmoothing);
				pval(pval <= 0.f) = TV.APLSsmoothing;
				pval = af::sqrt(pval);
				apu1 = (f - (apu * epsilon(af::span, 0))) / pval;
				apu2 = (g - (apu * epsilon(af::span, 1))) / pval;
				apu4 = (f - (apu * epsilon(af::span, 0)) + g - (apu * epsilon(af::span, 1))) / pval;
			}
		}
	}
	// If anatomical prior is not used
	else {
		if (TVtype == 4U) {
			const af::array ff = f / (af::abs(f) + 1e-8f);
			const af::array gg = g / (af::abs(g) + 1e-8f);
			if (Nz > 1) {
				const af::array hh = h / af::abs(h);
				if (TV.SATVPhi == 0.f) {
					apu1 = ff;
					apu2 = gg;
					apu3 = hh;
				}
				else {
					apu1 = ff - ff / (af::abs(f) / TV.SATVPhi + 1.f);
					apu2 = gg - gg / (af::abs(g) / TV.SATVPhi + 1.f);
					apu3 = hh - hh / (af::abs(h) / TV.SATVPhi + 1.f);
				}
			}
			else {
				if (TV.SATVPhi == 0.f) {
					apu1 = ff;
					apu2 = gg;
				}
				else {
					apu1 = ff - ff / (af::abs(f) / TV.SATVPhi + 1.f);
					apu2 = gg - gg / (af::abs(g) / TV.SATVPhi + 1.f);
				}
			}
			//mexPrintf("apu1.dims(0) = %d\n", apu1.dims(0));
			//mexPrintf("apu1.dims(1) = %d\n", apu1.dims(1));
			//mexPrintf("f.dims(0) = %d\n", f.dims(0));
			//mexPrintf("f.dims(1) = %d\n", f.dims(1));
			//mexEvalString("pause(.0001);");
		}
		else {
			if (Nz > 1) {
				pval = af::sqrt(af::pow(f, 2.) + af::pow(g, 2.) + af::pow(h, 2.) + TV.TVsmoothing);
				apu1 = f / pval;
				apu2 = g / pval;
				apu3 = h / pval;
			}
			else {
				pval = af::sqrt(af::pow(f, 2.) + af::pow(g, 2.) + TV.TVsmoothing);
				apu1 = f / pval;
				apu2 = g / pval;
			}
		}
		if (Nz > 1) {
			apu4 = apu1 + apu2 + apu3;
		}
		else
			apu4 = apu1 + apu2;
	}
	apu1 = af::moddims(apu1, Nx, Ny, Nz);
	apu2 = af::moddims(apu2, Nx, Ny, Nz);
	if (Nz > 1)
		apu3 = af::moddims(apu3, Nx, Ny, Nz);
	apu4 = af::moddims(apu4, Nx, Ny, Nz);
	// Derivatives
	apu1 = af::shift(apu1, 1);
	apu2 = af::shift(apu2, 0, 1);
	if (Nz > 1) {
		apu3 = af::shift(apu3, 0, 0, 1);
		gradi = apu4 - apu1 - apu2 - apu3;
	}
	else
		gradi = apu4 - apu1 - apu2;
	gradi = af::flat(gradi);
	gradi = gradi + 2.f * TV.tau * af::min<float>(af::flat(ima));
	//mexPrintf("gradi.summa = %f\n", af::sum<float>(flat(gradi)));
	//mexEvalString("pause(.0001);");

	return gradi;
}

af::array Quadratic_prior(const af::array& im, const uint32_t Ndx, const uint32_t Ndy, const uint32_t Ndz, const uint32_t Nx, const uint32_t Ny, const uint32_t Nz,
	const af::array& weights_quad)
{
	const af::array apu_pad = padding(im, Nx, Ny, Nz, Ndx, Ndy, Ndz);
	af::array grad;
	//af::array weights = weights_quad;
	if (Ndz == 0 || Nz == 1) {
		grad = af::convolve2(apu_pad, weights_quad);
		grad = grad(af::seq(Ndx, Nx + Ndx - 1), af::seq(Ndy, Ny + Ndy - 1), af::span);
	}
	else {
		grad = af::convolve3(apu_pad, weights_quad);
		grad = grad(af::seq(Ndx, Nx + Ndx - 1), af::seq(Ndy, Ny + Ndy - 1), af::seq(Ndz, Nz + Ndz - 1));
	}
	grad = af::flat(grad);
	return grad;
}

af::array Huber_prior(const af::array& im, const uint32_t Ndx, const uint32_t Ndy, const uint32_t Ndz, const uint32_t Nx, const uint32_t Ny, const uint32_t Nz,
	const af::array& weights_huber, const float delta)
{
	af::array grad = Quadratic_prior(im, Ndx, Ndy, Ndz, Nx, Ny, Nz, weights_huber);
	//if (af::sum<dim_t>(delta >= af::abs(af::flat(grad))) == grad.elements() && af::sum<int>(af::flat(grad)) != 0)
	//	mexPrintf("Delta value of Huber prior larger than all the pixel difference values\n");
	grad(grad > delta) = delta;
	grad(grad < -delta) = -delta;
	return grad;
}

void computePminusCG(const bool useF, const bool sparseF, const bool useKineticModel, const uint8_t complexType, array& temp, const array& F,
	const array& Pplus, const array& Pplusi, const bool complexF, const array& Fi) {
	if (useF) {
		if (sparseF) {
			if (useKineticModel) {
				if (complexType == 2) {
					temp(seq(F.dims(0), end)) = matmul(F, temp);
					temp = matmul(Pplusi, Pplusi.T(), temp);
					temp(seq(0, F.dims(0) - 1)) = matmul(F, temp);
				}
				else {
					temp(seq(F.dims(0), end)) = matmul(F, temp);
					temp = matmul(Pplus, Pplus.T(), temp);
					temp(seq(0, F.dims(0) - 1)) = matmul(F, temp);
				}
			}
			else {
				if (complexType == 2) {
					if (complexF)
						temp = matmul(Fi, matmul(Pplusi, Pplusi.T(), matmul(Fi, temp, AF_MAT_TRANS)));
					else
						temp = matmul(F, matmul(Pplusi, Pplusi.T(), matmul(F, temp, AF_MAT_TRANS)));
				}
				else {
					temp = matmul(F, matmul(Pplus, Pplus.T(), matmul(F, temp, AF_MAT_TRANS)));
				}
			}
		}
		else
			if (complexType == 2) {
				if (complexF)
					temp = matmul(Fi, matmul(Pplusi, Pplusi.T(), Fi.T(), temp));
				else
					temp = matmul(F, matmul(Pplusi, Pplusi.T(), F.T(), temp));
			}
			else
				temp = matmul(F, matmul(Pplus, Pplus.T(), F.T(), temp));
	}
	else
		if (complexType == 2)
			temp = matmul(Pplusi, Pplusi.T(), temp);
		else
			temp = matmul(Pplus, Pplus.T(), temp);
}

array getMeas(const uint32_t regularization, const uint64_t window, const array& m0, const uint64_t nMeas, const uint32_t NN, const uint64_t Nm, const uint64_t tt, 
	const uint8_t complexType, const uint64_t hnU, const array& L, const bool complex = false) {
	array SS;
	if (complexType == 3) {
		if (regularization == 1)
			SS = join(0, join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), L), join(0, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), L));
		else
			SS = join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))));
	}
	else {
		if (complex) {
			if (regularization == 1)
				SS = join(0, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), L);
			else
				SS = imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))));
		}
		else {
			if (regularization == 1)
				SS = join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), L);
			else
				SS = real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))));
		}
	}
	return SS;
}

void computeInnovation(const uint32_t regularization, array& SS, const uint64_t window, const array& m0, const uint64_t nMeas, const uint32_t NN,
	const uint64_t Nm, const uint64_t tt, const array& S, const array& xt, const array& L, const std::vector<array>& Si, const uint8_t complexType, const uint64_t hnU) {
	if (complexType == 3) {
		SS = getMeas(regularization, window, m0, nMeas, NN, Nm, tt, complexType, hnU, L) -
			join(0, matmul(S, xt(seq(0, xt.dims(0) / 2 - 1))) - matmul(Si[tt % hnU], xt(seq(xt.dims(0) / 2, end))),
				matmul(Si[tt % hnU], xt(seq(0, xt.dims(0) / 2 - 1))) + matmul(S, xt(seq(xt.dims(0) / 2, end))));
	}
	else {
		SS = getMeas(regularization, window, m0, nMeas, NN, Nm, tt, complexType, hnU, L) - matmul(S, xt);
	}
}

// Computes the KF denoising
void computeDenoising(array& xt, const uint64_t imDim, const int64_t oo, const uint32_t NN, const uint8_t complexType, const uint32_t regularization,
	const array& Pplus, const array& Pplusi, const uint32_t prior, const uint32_t nIter, const uint32_t Nx, const uint32_t Ny, const uint32_t DimZ,
	const TVdata& TV, const TVdata& TVi, const uint32_t Ndx, const uint32_t Ndy, const uint32_t Ndz, const float gamma, const float beta, const float betac,
	const float huberDelta, const array& weightsHuber, const array& weightsQuad, const array& LL, const bool complexRef, const array& Li, const TGVdata TGV,
	const int32_t Type = 0, const array Pr = constant(0.f, 1, 1)) {

	array vr, vi, gradr, gradi, Papu, Papui;
	if (complexType <= 2)
		vr = real(xt(seq(0, imDim - 1), oo + 1, NN));
	if (complexType == 1 || complexType == 2)
		vi = imag(xt(seq(0, imDim - 1), oo + 1, NN));
	else if (complexType == 3) {
		vr = real(xt(seq(0, imDim - 1), oo + 1, NN));
		vi = imag(xt(seq(0, imDim - 1), oo + 1, NN));
	}
	if (regularization == 4 && Type == 0) {
		Papu = inverse(Pplus);
		if (complexType == 2)
			Papui = inverse(Pplusi);
	}
	if ((regularization == 3 && prior < 8) || regularization == 4) {
		for (uint32_t ii = 0; ii < nIter; ii++) {
			if (regularization == 3) {
				if (prior < 7U) {
					if (prior == 1) {
						gradr = TVprior(Nx, Ny, DimZ, TV, vr, 1, NN);
						if (complexType > 0)
							gradi = TVprior(Nx, Ny, DimZ, TV, vi, 1, NN);
					}
					else if (prior == 2) {
						gradr = TVprior(Nx, Ny, DimZ, TV, vr, 2, NN);
						if (complexType > 0)
							gradi = TVprior(Nx, Ny, DimZ, TV, vi, 2, NN);
					}
					else if (prior == 3) {
						gradr = TVprior(Nx, Ny, DimZ, TV, vr, 3, NN);
						if (complexType > 0)
							gradi = TVprior(Nx, Ny, DimZ, TV, vi, 3, NN);
					}
					else if (prior == 4) {
						gradr = TVprior(Nx, Ny, DimZ, TV, vr, 4, NN);
						if (complexType > 0)
							gradi = TVprior(Nx, Ny, DimZ, TV, vi, 4, NN);
					}
					//else if (prior == 5) {
					//	gradr = Quadratic_prior(vr, Ndx, Ndy, Ndz, Nx, Ny, DimZ, weightsQuad);
					//	if (complexType > 0)
					//		gradi = Quadratic_prior(vi, Ndx, Ndy, Ndz, Nx, Ny, DimZ, weightsQuad);
					//}
					//else if (prior == 6) {
					//	gradr = Huber_prior(vr, Ndx, Ndy, Ndz, Nx, Ny, DimZ, weightsHuber, huberDelta);
					//	if (complexType > 0)
					//		gradi = Huber_prior(vi, Ndx, Ndy, Ndz, Nx, Ny, DimZ, weightsHuber, huberDelta);
					//}
					//mexPrintf("gradr.summa = %f\n", af::sum<float>(flat(gradr)));
					//mexPrintf("gradr.min = %f\n", af::min<float>(abs(flat(gradr))));
					//mexEvalString("pause(.0001);");
					if (complexType == 3)
						if (Type == 0)
							vr -= (beta * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), gradr));
						else if (Type == 2)
							vr -= (beta * matmul(Pr, Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pr.T(), gradr));
						else if (Type == 3)
							vr -= (beta * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pplus(seq(0, Pplus.dims(0) / 2 - 1), span).T(), gradr));
						else
							vr -= (beta * gradr);
					else
						if (Type == 0)
							vr -= (beta * matmul(Pplus, gradr));
						else if (Type == 2)
							vr -= (beta * matmul(Pr, Pplus, Pr.T(), gradr));
						else if (Type == 3)
							vr -= (beta * matmul(Pplus, Pplus.T(), gradr));
						else
							vr -= (beta * gradr);
					if (complexType == 1)
						if (Type == 0)
							vi -= (betac * matmul(Pplus, gradi));
						else if (Type == 2)
							vi -= (betac * matmul(Pr, Pplus, Pr.T(), gradi));
						else if (Type == 3)
							vi -= (betac * matmul(Pplus, Pplus.T(), gradi));
						else
							vi -= (betac * gradi);
					else if (complexType == 2)
						if (Type == 0)
							vi -= (betac * matmul(Pplusi, gradi));
						else if (Type == 2)
							vi -= (betac * matmul(Pr, Pplusi, Pr.T(), gradi));
						else if (Type == 3)
							vi -= (betac * matmul(Pplusi, Pplusi.T(), gradi));
						else
							vi -= (betac * gradi);
					else if (complexType == 3)
						if (Type == 0)
							vi -= (betac * matmul(Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)), gradi));
						else if (Type == 2)
							vi -= (betac * matmul(Pr, Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pr.T(), gradi));
						else if (Type == 3)
							vi -= (betac * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pplus(seq(0, Pplus.dims(0) / 2 - 1), span).T(), gradi));
						else
							vi -= (betac * gradi);
				}
				else {
					if (complexType == 3) {
						if (Type == 0)
							vr -= (beta * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), matmul(LL, vr)));
						else if (Type == 2)
							vr -= (beta * matmul(Pr, Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), Pr.T(), matmul(LL, vr)));
						else if (Type == 3)
							vr -= (beta * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pplus(seq(0, Pplus.dims(0) / 2 - 1), span).T(), matmul(LL, vr)));
						else
							vr -= (beta * matmul(LL, vr));
					}
					else {
						if (Type == 0)
							vr += (beta * matmul(real(Pplus), matmul(LL, vr)));
						else if (Type == 2)
							vr -= (beta * matmul(Pr, Pplus, Pr.T(), matmul(LL, vr)));
						else if (Type == 3)
							vr -= (beta * matmul(Pplus, Pplus.T(), matmul(LL, vr)));
						else
							vr += (beta * matmul(LL, vr));
					}
					if (complexType == 1) {
						if (complexRef) {
							if (Type == 0)
								vi -= (betac * matmul(Pplus, matmul(Li, vi)));
							else if (Type == 2)
								vi -= (betac * matmul(Pr, Pplus, Pr.T(), matmul(Li, vi)));
							else if (Type == 3)
								vi -= (betac * matmul(Pplus, Pplus.T(), matmul(Li, vi)));
							else
								vi -= (betac * matmul(Li, vi));
						}
						else {
							if (Type == 0)
								vi += (betac * matmul(Pplus, matmul(LL, vi)));
							else if (Type == 2)
								vi -= (betac * matmul(Pr, Pplus, Pr.T(), matmul(LL, vi)));
							else if (Type == 3)
								vi -= (betac * matmul(Pplus, Pplus.T(), matmul(LL, vi)));
							else
								vi += (betac * matmul(LL, vi));
						}
					}
					else if (complexType == 2) {
						if (complexRef)
							if (Type == 0)
								vi -= (betac * matmul(Pplusi, matmul(Li, vi)));
							else if (Type == 2)
								vi -= (betac * matmul(Pr, Pplusi, Pr.T(), matmul(Li, vi)));
							else if (Type == 3)
								vi -= (betac * matmul(Pplusi, Pplusi.T(), matmul(Li, vi)));
							else
								vi -= (betac * matmul(Li, vi));
						else
							if (Type == 0)
								vi -= (betac * matmul(Pplusi, matmul(LL, vi)));
							else if (Type == 2)
								vi -= (betac * matmul(Pr, Pplusi, Pr.T(), matmul(LL, vi)));
							else if (Type == 3)
								vi -= (betac * matmul(Pplusi, Pplusi.T(), matmul(LL, vi)));
							else
								vi -= (betac * matmul(LL, vi));
					}
					else if (complexType == 3) {
						if (complexRef)
							if (Type == 0)
								vi -= (betac * matmul(Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)), matmul(Li, vi)));
							else if (Type == 2)
								vi -= (betac * matmul(Pr, Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), Pr.T(), matmul(Li, vi)));
							else if (Type == 3)
								vi -= (betac * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pplus(seq(0, Pplus.dims(0) / 2 - 1), span).T(), matmul(Li, vi)));
							else
								vi -= (betac * matmul(Li, vi));
						else
							if (Type == 0)
								vi -= (betac * matmul(Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)), matmul(LL, vi)));
							else if (Type == 2)
								vi -= (betac * matmul(Pr, Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), Pr.T(), matmul(LL, vi)));
							else if (Type == 3)
								vi -= (betac * matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), span), Pplus(seq(0, Pplus.dims(0) / 2 - 1), span).T(), matmul(LL, vi)));
							else
								vi -= (betac * matmul(LL, vi));
					}
				}
			}
			else {
				const array apuX = xt(seq(0, imDim - 1), oo + 1, NN);
				if (TV.TV_use_anatomical) {
					if (complexType == 3)
						if (Type == 0)
							vr = vr * exp(-gamma * (matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), vr - real(apuX)) + beta * matmul((LL), vr - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
						else
							vr = vr * exp(-gamma * (vr - real(apuX) + beta * matmul((LL), vr - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
					else
						if (Type == 0)
							vr = vr * exp(-gamma * (matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), vr - real(apuX)) + beta * matmul((LL), vr - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
						else
							vr = vr * exp(-gamma * (vr - real(apuX) + beta * matmul((LL), vr - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
					if (complexType == 1)
						if (complexRef)
							if (Type == 0)
								vi = vi * exp(-gamma * (matmul((Papu), vi - imag(apuX)) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
							else
								vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
						else
							if (Type == 0)
								vi = vi * exp(-gamma * (matmul((Papu), vi - imag(apuX)) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
							else
								vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
					else if (complexType == 2)
						if (complexRef)
							if (Type == 0)
								vi = vi * exp(-gamma * (matmul((Papui), vi - imag(apuX)) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
							else
								vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
						else
							if (Type == 0)
								vi = vi * exp(-gamma * (matmul((Papui), vi - imag(apuX)) + betac * matmul(LL, vi - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
							else
								vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul(LL, vi - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
					else if (complexType == 3)
						if (complexRef)
							if (Type == 0)
								vi = vi * exp(-gamma * (matmul(Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)), vi - imag(apuX)) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
							else
								vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul((LL), vi - flat(TVi.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
						else
							if (Type == 0)
								vi = vi * exp(-gamma * (matmul(Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)), vi - imag(apuX)) + betac * matmul(LL, vi - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
							else
								vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul(LL, vi - flat(TV.reference_image(seq(imDim * NN, imDim * (NN + 1) - 1))))));
				}
				else {
					if (complexType == 3)
						if (Type == 0)
							vr = vr * exp(-gamma * (matmul(Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1)), vr - real(apuX)) + beta * matmul((LL), vr)));
						else
							vr = vr * exp(-gamma * (vr - real(apuX) + beta * matmul((LL), vr)));
					else
						if (Type == 0)
							vr = vr * exp(-gamma * (matmul(real(Papu), vr - real(apuX)) + beta * matmul((LL), vr)));
						else
							vr = vr * exp(-gamma * (vr - real(apuX) + beta * matmul((LL), vr)));
					if (complexType == 1)
						if (Type == 0)
							vi = vi * exp(-gamma * (matmul(real(Papu), vi - imag(apuX)) + betac * matmul(LL, vi)));
						else
							vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul(LL, vi)));
					else if (complexType == 2)
						if (Type == 0)
							vi = vi * exp(-gamma * (matmul((Papui), vi - imag(apuX)) + betac * matmul(LL, vi)));
						else
							vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul(LL, vi)));
					else if (complexType == 3)
						if (Type == 0)
							vi = vi * exp(-gamma * (matmul(Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)), vi - imag(apuX)) + betac * matmul(LL, vi)));
						else
							vi = vi * exp(-gamma * (vi - imag(apuX) + betac * matmul(LL, vi)));
				}
			}
		}
	}
	else {
		vr = TGVDenoising(Nx, Ny, DimZ, TGV.lambda1, TGV.lambda2, TGV.rho, TGV.prox, vr, nIter);
		if (complexType > 0)
			vi = TGVDenoising(Nx, Ny, DimZ, TGV.lambda1, TGV.lambda2, TGV.rho, TGV.prox, vi, nIter);
	}
	if (complexType == 0)
		xt(seq(0, imDim - 1), oo + 1, NN) = flat(vr);
	else
		xt(seq(0, imDim - 1), oo + 1, NN) = complex(flat(vr), flat(vi));
}

// Tranposed solve
array solveT(const array& A, const array& B) {
	return transpose(solve(transpose(A), B));
}

// Tranposed solve
array solveTT(const array& A, const array& B) {
	return transpose(solve(transpose(A), transpose(B)));
}

// Load the required system matrix component for the current time step from system memory
void loadSystemMatrix(const bool storeData, const bool sparseS, const uint64_t tt, const uint64_t window, const uint64_t hn, const uint64_t hnU,
	const uint64_t Nm, const bool complexS, const size_t lSize, const uint8_t complexType, const uint32_t regularization, const uint64_t imDimN,
	const size_t* sCol, const std::vector<float>& S1, const std::vector<float>& S2, array& Svalues, array& Svaluesi, array& Srow, array& Scol,
	const std::vector<int32_t>& sCols, const std::vector<int32_t>& sRows, const array& Lvalues, const array& Lcol, const array& LL, const float* SS3,
	const float* SS4, std::vector<array>& S, std::vector<array>& Si) {
	// If the system matrix components were not stored and the matrix is sparse
	if (!storeData && sparseS && tt > 0ULL) {
		// Specific possible sliding window cases
		if (window == 1 || (window > 1 && tt % hn <= hn - window)) {
			const array jelppi = seq(sCol[Nm * (tt % hn)], sCol[Nm * ((tt % hn) + 1) + Nm * (window - 1ULL)] - 1);
			Svalues = array(jelppi.dims(0), &S1[sCol[Nm * (tt % hn)]], afHost);
			if (complexS && (complexType == 3 || complexType == 2))
				Svaluesi = array(jelppi.dims(0), &S2[sCol[Nm * (tt % hn)]], afHost);
			Srow = array((Nm * window + lSize) + 1, &sCols[(Nm * window + lSize) * (tt % hn) + (tt % hn)], afHost);
			Scol = array(jelppi.dims(0), &sRows[sCol[Nm * (tt % hn)]], afHost);
			//mexPrintf("Svalues.dims(0) = %d\n", Svalues.dims(0));
			//mexPrintf("jelppi.dims(0) = %d\n", jelppi.dims(0));
			//mexPrintf("Srow.dims(0) = %d\n", Srow.dims(0));
			//mexPrintf("Scol.dims(0) = %d\n", Scol.dims(0));
			//mexPrintf("sCol[Nm * tt] = %d\n", sCol[Nm * tt]);
			//mexPrintf("sCol[Nm * (tt + 1) + Nm * (window - 1ULL)] = %d\n", sCol[Nm * (tt + 1) + Nm * (window - 1ULL)]);
		}
		else {
			//mexPrintf("tt = %d\n", tt);
			//mexEvalString("pause(.0001);");
			//mexPrintf("Nm * (window - hn - (tt % hn)) = %d\n", Nm * (window - (hn - (tt % hn))));
			//mexPrintf("sCol[Nm * (window - hn - (tt % hn))] = %d\n", sCol[Nm * (window - (hn - (tt % hn)))]);
			//mexPrintf("sCol[Nm * (tt % hn)] = %d\n", sCol[Nm * (tt % hn)]);
			//mexPrintf("sRows[sCol[Nm * (tt % hn)]] = %d\n", sRows[sCol[Nm * (tt % hn)]]);
			//mexPrintf("(Nm * window + lSize) * (tt % hn) + (tt % hn) = %d\n", (Nm * window + lSize) * (tt % hn) + (tt % hn));
			//mexEvalString("pause(.0001);");
			const array jelppi = seq(sCol[Nm * (tt % hn)], sCol[Nm * hn] - 1);
			const array jelppi2 = seq(sCol[0], sCol[Nm * (window - (hn - (tt % hn)))] - 1);
			Svalues = join(0, array(jelppi.dims(0), &S1[sCol[Nm * (tt % hn)]], afHost), array(jelppi2.dims(0), &S1[0], afHost));
			if (complexS && (complexType == 3 || complexType == 2))
				Svaluesi = join(0, array(jelppi.dims(0), &S2[sCol[Nm * (tt % hn)]], afHost), array(jelppi2.dims(0), &S2[0], afHost));
			Srow = array((Nm * window + lSize) + 1, &sCols[(Nm * window + lSize) * (tt % hn) + (tt % hn)], afHost);
			Scol = join(0, array(jelppi.dims(0), &sRows[sCol[Nm * (tt % hn)]], afHost), array(jelppi2.dims(0), &sRows[0], afHost));
		}
		// AKF
		if (regularization == 1)
			S[0] = sparse(lSize + Nm * window, imDimN, join(0, Svalues, Lvalues), Srow, join(0, Scol, Lcol));
		else
			S[0] = sparse(Nm * window, imDimN, Svalues, Srow, Scol);
		if (complexS && (complexType == 3 || complexType == 2)) {
			if (regularization == 1)
				Si[0] = sparse(lSize + Nm * window, imDimN, join(0, Svaluesi, Lvalues), Srow, join(0, Scol, Lcol));
			else
				Si[0] = sparse(Nm * window, imDimN, Svaluesi, Srow, Scol);
		}
	}
	// If the system matrix components were not stored and the matrix is not sparse
	else if (!storeData && !sparseS && tt > 0ULL) {
		if (window == 1 || (window > 1 && tt < (hn - window + 1))) {
			// AKF
			if (regularization == 1) {
				S[0] = join(0, transpose(array(imDimN, Nm * window, &SS3[(tt % hn) * Nm])), dense(LL));
				if (complexS && (complexType == 3 || complexType == 2))
					Si[0] = join(0, transpose(array(imDimN, Nm * window, &SS4[(tt % hn) * Nm])), dense(LL));
			}
			else {
				S[0] = transpose(array(imDimN, Nm * window, &SS3[(tt % hn) * Nm * imDimN]));
				if (complexS && (complexType == 3 || complexType == 2))
					Si[0] = transpose(array(imDimN, Nm * window, &SS4[(tt % hn) * Nm * imDimN]));
			}
		}
		else {
			// AKF
			if (regularization == 1) {
				S[0] = join(0, join(0, transpose(array(imDimN, Nm * (hn - (tt % hn)), &SS3[(tt % hn) * Nm])), transpose(array(imDimN, Nm * (window - (hn - (tt % hn))), &SS3[0]))), dense(LL));
				if (complexS && (complexType == 3 || complexType == 2))
					Si[0] = join(0, join(0, transpose(array(imDimN, Nm * (hn - (tt % hn)), &SS4[(tt % hn) * Nm])), transpose(array(imDimN, Nm * (window - (hn - (tt % hn))), &SS4[0]))), dense(LL));
			}
			else {
				S[0] = join(0, transpose(array(imDimN, Nm * (hn - (tt % hn)), &SS3[(tt % hn) * Nm])), transpose(array(imDimN, Nm * (window - (hn - (tt % hn))), &SS3[0])));
				if (complexS && (complexType == 3 || complexType == 2))
					Si[0] = join(0, transpose(array(imDimN, Nm * (hn - (tt % hn)), &SS4[(tt % hn) * Nm])), transpose(array(imDimN, Nm * (window - (hn - (tt % hn))), &SS4[0])));
			}
		}
	}
	//mexPrintf("S[0].dims(0) = %d\n", S[0].dims(0));
	//mexPrintf("S[0].dims(1) = %d\n", S[0].dims(1));
	//mexEvalString("pause(.0001);");
}

// Compute the a prior covariance
// Used only if the state transition matrix F is used
void computePminus(const uint32_t algorithm, const bool sparseF, array& Pplus, const array& F, const uint64_t tt, const uint64_t sizeF, const uint64_t sizeQ,
	const array& Q, const bool sparseQ, const bool useKineticModel, const uint32_t NN, const array& KG, const std::vector<array>& S, const uint64_t hnU,
	const bool useF = false) {
	// Regular KF
	if (algorithm == 0) {
		//mexPrintf("F.dims(0) = %d\n", F.dims(0));
		//mexPrintf("F.dims(1) = %d\n", F.dims(1));
		//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
		//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
		//mexEvalString("pause(.0001);");
		if (sparseF) {
			if (useKineticModel) {
				Pplus(seq(0, Pplus.dims(0) / 2 - 1), span) = matmul(F, Pplus);
				Pplus(span, seq(0, Pplus.dims(0) / 2 - 1)) = transpose(matmul(F, transpose(Pplus)));
			}
			else
				Pplus = transpose(matmul(F, transpose(matmul(F, Pplus))));
		}
		else
			Pplus = matmul(F, Pplus, F.T());
	}
	// Information filter
	else if (algorithm == 1) {
		if (sparseQ) {
			if (sparseF) {
				Pplus = Q - transpose(matmul(Q, matmul(F, transpose(matmul(Q, matmul(F, inverse(Pplus + matmul(F,
					matmul(Q, dense(F)), AF_MAT_TRANS)))))), AF_MAT_TRANS));
			}
			else {
				Pplus = Q - matmul(solve(Pplus + matmul(F, matmul(Q, F), AF_MAT_TRANS), matmul(Q, F)),
					transpose(matmul(Q, F, AF_MAT_TRANS)));
			}
		}
		else {
			if (sparseF) {
				Pplus = batchFunc(Q(span, NN), batchFunc(pow(Q(span, NN), 2.), transpose(matmul(F, transpose(matmul(F, inverse(Pplus + matmul(F, batchFunc(Q(span, NN), dense(F), batchMul), AF_MAT_TRANS)))))), batchMul), batchMinus);
			}
			else {
				Pplus = batchFunc(Q(span, NN), batchFunc(matmulNT(batchFunc(Q(span, NN), solve(Pplus + matmulTN(F, batchFunc(F, Q(span, NN), batchMul)),
					F), batchMul), F), Q(span, NN), batchMul), batchMinus);
			}
		}
	}
	else if (algorithm == 2) {
		if (useF) {
			if (sparseF) {
				Pplus = transpose(matmul(F - transpose(matmul(S[tt % hnU], transpose(KG), AF_MAT_TRANS)), transpose(matmul(F, Pplus))));
			}
			else {
				Pplus = matmul(matmul(F, Pplus), F - transpose(matmul(S[tt % hnU], transpose(KG), AF_MAT_TRANS)), AF_MAT_TRANS);
			}
		}
		else {
			Pplus = Pplus - matmul(Pplus, matmul(S[tt % hnU], transpose(KG), AF_MAT_TRANS));
		}
		if (sparseQ)
			Pplus = Pplus + Q;
		else
			Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q(span, NN);
	}
}

// Compute the one-step KF
void oneStepKF(array& HH, const std::vector<array>& S, const uint64_t tt, const uint64_t hnU, array& Pplus, const bool sparseR, const uint64_t sizeR,
	const std::vector<array>& R, array& KG, const bool useF, const std::vector<array>& F, const std::vector<array>& Fi, const uint64_t sizeF, const bool sparseF,
	const float RR, const uint32_t regularization, const bool useKineticModel, const bool complexF = false) {
	HH = transpose(matmul(S[tt % hnU], transpose(matmul(S[tt % hnU], Pplus))));
	if (sparseR)
		HH = HH + R[tt % sizeR];
	else {
		if (regularization == 1)
			HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR);
		else
			HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + R[tt % sizeR];
	}
	KG = transpose(solve(transpose(HH), matmul(S[tt % hnU], transpose(Pplus))));
	if (useF) {
		if (complexF)
			KG = matmul(Fi[tt % sizeF], KG);
		else
			//if (useKineticModel)
			//	KG(seq(0, F[tt % sizeF].dims(0) - 1), span) = matmul(F[tt % sizeF], KG);
			//else
			KG = matmul(F[tt % sizeF], KG);
	}
}

void computeInnovationCov(array& HH, const bool sparseR, const std::vector<array>& R, const array& Pplus, const std::vector<array>& S,
	const uint64_t tt, const uint64_t hnU, const uint64_t sizeR, const uint64_t ind, const float RR, const uint32_t regularization, const uint8_t complexType,
	const std::vector<array>& Si, const bool complexS = false, const int type = 0) {
	if (complexType == 3) {
		HH = transpose(join(0, matmul(S[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) - matmul(Si[ind], Pplus(seq(Pplus.dims(0) / 2, end), span)),
			matmul(Si[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) + matmul(S[ind], Pplus(seq(Pplus.dims(0) / 2, end), span))));
	}
	else
		if (complexS)
			if (type == 0)
				HH = (matmul(Si[ind], (Pplus)));
			else
				HH = matmulNT(matmul(Si[ind], (Pplus)), Pplus);
		else
			if (type == 0)
				HH = (matmul(S[ind], (Pplus)));
			else
				HH = matmulNT(matmul(S[ind], (Pplus)), Pplus);
	if (complexType == 3) {
		HH = transpose(join(0, matmul(S[ind], HH(seq(0, HH.dims(0) / 2 - 1), span)) - matmul(Si[ind], HH(seq(HH.dims(0) / 2, end), span)),
			matmul(Si[ind], HH(seq(0, HH.dims(0) / 2 - 1), span)) + matmul(S[ind], HH(seq(HH.dims(0) / 2, end), span))));
	}
	else
		if (complexS)
			HH = transpose(matmul(Si[ind], transpose(HH)));
		else
			HH = transpose(matmul(S[ind], transpose(HH)));
	if (sparseR)
		HH = HH + R[tt % sizeR];
	else {
		if (regularization == 1) {
			if (complexType == 3)
				HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, Pplus.dims(0), 1) * RR,
					R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, Pplus.dims(0), 1) * RR);
			else
				HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR);
		}
		else
			HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + R[tt % sizeR];
	}
}

// Compute the Kalman gain
void computeKG(const uint32_t algorithm, array& KG, array& HH, const bool sparseR, const std::vector<array>& R, const array& Pplus, const std::vector<array>& S,
	const uint64_t tt, const uint64_t hnU, const uint64_t sizeR, const uint64_t ind, const float RR, const uint32_t regularization, const uint8_t complexType,
	const std::vector<array>& Si, const array& P = constant(0.f, 1), const array& SS = constant(0.f, 1)) {
	if (algorithm == 0 || algorithm == 3) {
		if (algorithm == 3) {
			if (complexType == 3) {
				KG = transpose(matmulNT(join(0, matmul(S[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) - matmul(Si[ind], Pplus(seq(Pplus.dims(0) / 2, end), span)),
					matmul(Si[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) + matmul(S[ind], Pplus(seq(Pplus.dims(0) / 2, end), span))), Pplus));
			}
			else
				KG = matmulNT(matmul(S[ind], Pplus), Pplus);
		}
		else {
			//mexPrintf("ind = %d\n", ind);
			//mexEvalString("pause(.0001);");
			//const array apu = join(0, join(1, S[ind], -1.f * Si[ind]), join(1, Si[ind], S[ind]));
			if (complexType == 3) {
				KG = transpose(join(0, matmul(S[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) - matmul(Si[ind], Pplus(seq(Pplus.dims(0) / 2, end), span)),
					matmul(Si[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) + matmul(S[ind], Pplus(seq(Pplus.dims(0) / 2, end), span))));
				//KG = join(0, join(1, matmul(S[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1))) - matmul(Si[ind], Pplus(seq(Pplus.dims(0) / 2, end), seq(0, Pplus.dims(0) / 2 - 1))),
				//	matmul(S[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(Pplus.dims(0) / 2, end))) - matmul(Si[ind], Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)))), 
				//	join(1, matmul(Si[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1))) + matmul(S[ind], Pplus(seq(Pplus.dims(0) / 2, end), seq(0, Pplus.dims(0) / 2 - 1))),
				//		matmul(Si[ind], Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(Pplus.dims(0) / 2, end))) + matmul(S[ind], Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)))));
				//KG = (matmul(apu, (Pplus)));
			}
			else
				KG = (matmul(S[ind], (Pplus)));
		}
		mexPrintf("KG.dims(0) = %d\n", KG.dims(0));
		mexPrintf("KG.dims(1) = %d\n", KG.dims(1));
		mexPrintf("KG.summa = %f\n", af::sum<float>(flat(KG)));
		mexEvalString("pause(.0001);");
		if (complexType == 3) {
			HH = transpose(join(0, matmul(S[ind], KG(seq(0, KG.dims(0) / 2 - 1), span)) - matmul(Si[ind], KG(seq(KG.dims(0) / 2, end), span)),
				matmul(Si[ind], KG(seq(0, KG.dims(0) / 2 - 1), span)) + matmul(S[ind], KG(seq(KG.dims(0) / 2, end), span))));
			KG = transpose(KG);
			//HH = transpose(join(0, join(1, matmul(S[ind], transpose(KG(seq(0, KG.dims(0) / 2 - 1), seq(0, KG.dims(1) / 2 - 1)))) - matmul(Si[ind], transpose(KG(seq(0, KG.dims(0) / 2 - 1), seq(KG.dims(1) / 2, end)))),
			//	matmul(S[ind], transpose(KG(seq(KG.dims(0) / 2, end), seq(0, KG.dims(1) / 2 - 1)))) - matmul(Si[ind], transpose(KG(seq(KG.dims(0) / 2, end), seq(KG.dims(1) / 2, end))))),
			//	join(1, matmul(Si[ind], transpose(KG(seq(0, KG.dims(0) / 2 - 1), seq(0, KG.dims(1) / 2 - 1)))) + matmul(S[ind], transpose(KG(seq(KG.dims(0) / 2, end), seq(0, KG.dims(1) / 2 - 1)))),
			//		matmul(Si[ind], transpose(KG(seq(KG.dims(0) / 2, end), seq(0, KG.dims(1) / 2 - 1)))) + matmul(S[ind], transpose(KG(seq(KG.dims(0) / 2, end), seq(KG.dims(1) / 2, end)))))));
			//HH = transpose(matmul(apu, transpose(KG)));
		}
		else
			HH = transpose(matmul(S[ind], transpose(KG)));
		mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
		mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
		mexPrintf("HH.summa = %f\n", af::sum<float>(flat(HH)));
		mexEvalString("pause(.0001);");
		if (sparseR)
			HH = HH + R[tt % sizeR];
		else {
			if (regularization == 1) {
				if (complexType == 3)
					HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, Pplus.dims(0), 1) * RR,
						R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, Pplus.dims(0), 1) * RR);
				else
					HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR);
			}
			else
				HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + R[tt % sizeR];
		}
		//eval(KG);
		//eval(HH);
		//mexPrintf("KG.dims(0) = %d\n", KG.dims(0));
		//mexPrintf("KG.dims(1) = %d\n", KG.dims(1));
		mexPrintf("HH1.dims(0) = %d\n", HH.dims(0));
		mexPrintf("HH1.dims(1) = %d\n", HH.dims(1));
		//mexPrintf("HH.summa = %f\n", af::sum<float>(flat(HH)));
		mexEvalString("pause(.0001);");
		KG = transpose(solve(HH, (KG)));
		mexPrintf("KG1.dims(0) = %d\n", KG.dims(0));
		mexPrintf("KG1.dims(1) = %d\n", KG.dims(1));
		mexEvalString("pause(.0001);");
	}
	else if (algorithm == 1) {
		if (SS.dims(0) > 1) {
			if (sparseR)
				if (complexType == 3) {
					KG = matmul(R[tt % sizeR], SS);
					KG = matmul(Pplus, vecmul3(S, Si, SS, ind, true));
				}
				else
					KG = matmul(Pplus, matmul(S[tt % hnU], matmul(R[tt % sizeR], SS), AF_MAT_TRANS));
			else {
				if (complexType == 3) {
					if (regularization == 1)
						KG = join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR) * SS;
					else
						KG = R[tt % sizeR] * SS;
					KG = matmul(Pplus, vecmul3(S, Si, SS, ind, true));
				}
				else
					if (regularization == 1)
						KG = matmul(Pplus, matmul(S[tt % hnU], join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR) * SS, AF_MAT_TRANS));
					else
						KG = matmul(Pplus, matmul(S[tt % hnU], R[tt % sizeR] * SS, AF_MAT_TRANS));
				//KG = batchFunc(transpose(R[tt % sizeR]), transpose(matmul(S[tt % hnU], transpose(Pplus))), batchMul);
			}
		}
		else {
			if (sparseR)
				if (complexType == 3) {
					KG = transpose(matmul(R[tt % sizeR], matmul3(S, Si, Pplus, ind)));
				}
				else
					KG = transpose(matmul(R[tt % sizeR], matmul(S[ind], Pplus)));
			else {
				if (complexType == 3) {
					if (regularization == 1)
						KG = transpose(tile(join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, S[ind].dims(0)) * matmul3(S, Si, Pplus, ind));
					else
						KG = transpose(tile(R[tt % sizeR], 1, S[ind].dims(0)) * matmul3(S, Si, Pplus, ind));
				}
				else
					if (regularization == 1)
						KG = transpose(tile(join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, S[ind].dims(0)) * matmul(S[ind], Pplus));
					else
						KG = transpose(tile(R[tt % sizeR], 1, S[ind].dims(0)) * matmul(S[ind], Pplus));
				//KG = batchFunc(transpose(R[tt % sizeR]), transpose(matmul(S[tt % hnU], transpose(Pplus))), batchMul);
			}
		}
		//mexPrintf("KG.dims(0) = %d\n", KG.dims(0));
		//mexEvalString("pause(.0001);");
	}
}

array augmentedR(array& R, const uint32_t regularization, const float RR, const uint64_t dim, const uint8_t complexType) {
	array RA;
	if (regularization == 1) {
		if (complexType == 3)
			RA = join(0, R(seq(0, R.dims(0) / 2 - 1)), constant(1.f, dim, 1) * RR, R(seq(R.dims(0) / 2, end)), constant(1.f, dim, 1) * RR);
		else
			RA = join(0, R, constant(1.f, dim, 1) * RR);
		return RA;
	}
	else
		return R;
}

array matrixInversionP(const bool sparseQ, const bool useF, const bool complexF, const uint8_t complexType, const array& xTemp, const array& xTempi, 
	const std::vector<array>& Q, const std::vector<array>& Qi, const std::vector<array>& F, const std::vector<array>& Fi, const uint32_t NN, 
	const int64_t qq, const int64_t ff, const array& X, const array& Xi, const bool complex = true, const bool nReal = false) {
	array Pq, Pqi, tempPq, apuQ, apuQi;
	if (sparseQ) {
		if (!nReal)
			apuQ = matmul(Q[qq], xTemp);
		if (complexType == 2 && complex)
			apuQi = matmul(Qi[qq], xTempi);
		else if (complexType == 1 && complex)
			apuQi = matmul(Q[qq], xTempi);
	}
	else {
		if (!nReal)
			apuQ = xTemp * Q[qq](span, NN);
		if (complexType == 2 && complex)
			apuQi = xTempi * Qi[qq](span, NN);
		else if (complexType == 1 && complex)
			apuQi = xTempi * Q[qq](span, NN);
	}
	//mexPrintf("apuQ.dims(0) = %d\n", apuQ.dims(0));
	//mexPrintf("apuQ.dims(1) = %d\n", apuQ.dims(1));
	//mexPrintf("apuQ.summa = %f\n", af::sum<float>(flat(apuQ)));
	//mexEvalString("pause(.0001);");
	if (useF) {
		if (!nReal)
			Pq = matmul(F[ff], apuQ, AF_MAT_TRANS);
		if (complexType == 2 && complex)
			if (complexF)
				Pqi = matmul(Fi[ff], apuQi, AF_MAT_TRANS);
			else
				Pqi = matmul(F[ff], apuQi, AF_MAT_TRANS);
		else if (complexType == 1 && complex)
			Pqi = matmul(F[ff], apuQi, AF_MAT_TRANS);
	}
	else {
		if (!nReal)
			Pq = apuQ;
		if ((complexType == 2 || complexType == 1) && complex)
			Pqi = apuQi;
	}
	//mexPrintf("X.dims(0) = %d\n", X.dims(0));
	//mexPrintf("X.dims(1) = %d\n", X.dims(1));
	//mexPrintf("X.summa = %f\n", af::sum<float>(flat(X)));
	//mexEvalString("pause(.0001);");
	if (!nReal)
		Pq = matmulTN(X, Pq);
	if ((complexType == 2 || (complexType == 1 && !nReal))  && complex)
		Pqi = matmulTN(Xi, Pqi);
	else if (complexType == 1 && complex)
		Pqi = matmulTN(X, Pqi);
	//mexPrintf("Pq.dims(0) = %d\n", Pq.dims(0));
	//mexPrintf("Pq.dims(1) = %d\n", Pq.dims(1));
	//mexPrintf("Pq.summa = %f\n", af::sum<float>(flat(Pq)));
	//mexEvalString("pause(.0001);");
	if (!nReal) {
		tempPq = X;
		if (useF)
			tempPq = matmul(F[ff], tempPq);
		if (sparseQ)
			tempPq = matmul(Q[qq], tempPq);
		else
			tempPq = tempPq * tile(Q[qq](span, NN), 1, tempPq.dims(1));
		if (useF)
			tempPq = matmul(F[ff], tempPq, AF_MAT_TRANS);
		//mexPrintf("tempPq.dims(0) = %d\n", tempPq.dims(0));
		//mexPrintf("tempPq.dims(1) = %d\n", tempPq.dims(1));
		//mexEvalString("pause(.0001);");
		tempPq = identity(X.dims(1), X.dims(1)) + matmulTN(X, tempPq);
		//mexPrintf("tempPq.dims(0) = %d\n", tempPq.dims(0));
		//mexPrintf("tempPq.dims(1) = %d\n", tempPq.dims(1));
		//mexEvalString("pause(.0001);");
		Pq = matmul(X, solve(tempPq, Pq));
	}
	if (complexType == 2 && complex) {
		tempPq = Xi;
		if (useF)
			if (complexF)
				tempPq = matmul(Fi[ff], tempPq);
			else
				tempPq = matmul(F[ff], tempPq);
		if (sparseQ)
			tempPq = matmul(Qi[qq], tempPq);
		else
			tempPq = tempPq * tile(Qi[qq](span, NN), 1, tempPq.dims(1));
		if (useF)
			if (complexF)
				tempPq = matmul(Fi[ff], tempPq, AF_MAT_TRANS);
			else
				tempPq = matmul(F[ff], tempPq, AF_MAT_TRANS);
		tempPq = identity(Xi.dims(1), Xi.dims(1)) + matmulTN(Xi, tempPq);
		Pqi = matmul(Xi, solve(tempPq, Pqi));
	}
	else if (complexType == 1 && complex) {
		tempPq = X;
		if (useF)
			tempPq = matmul(F[ff], tempPq);
		if (sparseQ)
			tempPq = matmul(Q[qq], tempPq);
		else
			tempPq = tempPq * tile(Q[qq](span, NN), 1, tempPq.dims(1));
		if (useF)
			tempPq = matmul(F[ff], tempPq, AF_MAT_TRANS);
		if (!nReal) {
			tempPq = identity(Xi.dims(1), Xi.dims(1)) + matmulTN(Xi, tempPq);
			Pqi = matmul(Xi, solve(tempPq, Pqi));
		}
		else {
			tempPq = identity(X.dims(1), X.dims(1)) + matmulTN(X, tempPq);
			Pqi = matmul(X, solve(tempPq, Pqi));
		}
	}
	if (useF) {
		if (!nReal)
			Pq = matmul(F[ff], Pq);
		if (complexType == 2 && complex)
			if (complexF)
				Pqi = matmul(Fi[ff], Pqi);
			else
				Pqi = matmul(F[ff], Pqi);
		else if (complexType == 1 && complex)
			Pqi = matmul(F[ff], Pqi);
	}
	if (sparseQ) {
		if (!nReal)
			Pq = matmul(Q[qq], Pq);
		if (complexType == 2 && complex)
			Pqi = matmul(Qi[qq], Pqi);
		else if (complexType == 1 && complex)
			Pqi = matmul(Q[qq], Pqi);
	}
	else {
		if (!nReal)
			Pq = Pq * Q[qq](span, NN);
		if (complexType == 2 && complex)
			Pqi = Pqi * Qi[qq](span, NN);
		else if (complexType == 1 && complex)
			Pqi = Pqi * Q[qq](span, NN);
	}
	if (!nReal)
		Pq = apuQ - Pq;
	//mexPrintf("Pq.dims(0) = %d\n", Pq.dims(0));
	//mexPrintf("Pq.dims(1) = %d\n", Pq.dims(1));
	//mexPrintf("Pq.summa = %f\n", af::sum<float>(flat(Pq)));
	//mexEvalString("pause(.0001);");
	if ((complexType == 2 || complexType == 1) && complex) {
		Pqi = apuQi - Pqi;
		if (!nReal)
			return join(1, Pq, Pqi);
		else
			return Pqi;
	}
	else
		return Pq;
}

void storeConsistency(const uint64_t tt, const uint64_t Nt, const uint64_t stepSize, array& eps, const array& SS, const array& HH, const int64_t cc, array& v,
	const bool sparseR, const uint32_t regularization, const array& Pplus, array& epsP, const bool computeBayesianP, const array& R, const float RR, const array& S,
	const uint8_t complexType, const std::vector<array>& Si, const uint64_t hnU, const bool computeP, const uint8_t type = 0) {
	if (!computeP) {
		if (tt < Nt - stepSize)
			if (type == 0)
				eps(cc) = matmulTN(SS, solve(HH, SS));
			else
				eps(cc) = matmul(SS.T(), HH, HH.T(), SS);
		v(span, cc) = SS;
	}
	else if (computeBayesianP && tt < Nt - stepSize && computeP) {
		array apuP = solve(HH, SS);
		if (sparseR)
			apuP = matmul(R, apuP);
		else {
			if (regularization == 1) {
				apuP = batchFunc(apuP, join(0, R, constant(1.f, Pplus.dims(0), 1) * RR), batchMul);
			}
			else
				apuP = batchFunc(apuP, R, batchMul);
		}
		array apuPP1;
		if (complexType == 3) {
			apuPP1 = transpose(join(0, join(1, matmul(S, Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1))) - matmul(Si[tt % hnU], Pplus(seq(Pplus.dims(0) / 2, end), seq(0, Pplus.dims(0) / 2 - 1))),
				matmul(S, Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(Pplus.dims(0) / 2, end))) - matmul(Si[tt % hnU], Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end)))),
				join(1, matmul(Si[tt % hnU], Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(0, Pplus.dims(0) / 2 - 1))) + matmul(S, Pplus(seq(Pplus.dims(0) / 2, end), seq(0, Pplus.dims(0) / 2 - 1))),
					matmul(Si[tt % hnU], Pplus(seq(0, Pplus.dims(0) / 2 - 1), seq(Pplus.dims(0) / 2, end))) + matmul(S, Pplus(seq(Pplus.dims(0) / 2, end), seq(Pplus.dims(0) / 2, end))))));
			apuPP1 = transpose(join(0, join(1, matmul(S, apuPP1(seq(0, apuPP1.dims(0) / 2 - 1), seq(0, apuPP1.dims(0) / 2 - 1))) - matmul(Si[tt % hnU], apuPP1(seq(apuPP1.dims(0) / 2, end), seq(0, apuPP1.dims(0) / 2 - 1))),
				matmul(S, apuPP1(seq(0, apuPP1.dims(0) / 2 - 1), seq(apuPP1.dims(0) / 2, end))) - matmul(Si[tt % hnU], apuPP1(seq(apuPP1.dims(0) / 2, end), seq(apuPP1.dims(0) / 2, end)))),
				join(1, matmul(Si[tt % hnU], apuPP1(seq(0, apuPP1.dims(0) / 2 - 1), seq(0, apuPP1.dims(0) / 2 - 1))) + matmul(S, apuPP1(seq(apuPP1.dims(0) / 2, end), seq(0, apuPP1.dims(0) / 2 - 1))),
					matmul(Si[tt % hnU], apuPP1(seq(0, apuPP1.dims(0) / 2 - 1), seq(apuPP1.dims(0) / 2, end))) + matmul(S, apuPP1(seq(apuPP1.dims(0) / 2, end), seq(apuPP1.dims(0) / 2, end))))));
		}
		else
			apuPP1 = transpose(matmul(S, transpose(matmul(S, Pplus))));
		if (sparseR)
			apuPP1 = matmul(R, apuPP1);
		else {
			if (regularization == 1) {
				apuPP1(seq(0, end, apuPP1.dims(0) + 1)) = apuPP1(seq(0, end, apuPP1.dims(0) + 1)) + join(0, R, constant(1.f, Pplus.dims(0), 1) * RR);
			}
			else
				apuPP1(seq(0, end, apuPP1.dims(0) + 1)) = apuPP1(seq(0, end, apuPP1.dims(0) + 1)) + R;
		}
		epsP(cc) = matmulTN(apuP, solve(apuPP1, apuP));
	}
}

void computeConsistencyTests(const int64_t cc, const uint64_t Nt, const uint64_t stepSize, const uint64_t initialSteps, const array& v, const uint64_t Nm,
	const array& eps, const array& epsP, const bool computeBayesianP, const uint8_t complexType, const array& vvi, const array& epsi, const array& epsPi) {
	//mexPrintf("cc = %d\n", cc);
	const uint64_t N = Nt - initialSteps - stepSize;
	//mexPrintf("N = %d\n", N);
	//mexPrintf("v.dims(0) = %d\n", v.dims(0));
	//mexPrintf("v.dims(1) = %d\n", v.dims(1));
	double chi1 = -1.96 + std::sqrt(static_cast<double>(2ULL * Nm * N - 1ULL));
	chi1 = (0.5 * chi1 * chi1) / static_cast<double>(N);
	double chi2 = 1.96 + std::sqrt(static_cast<double>(2ULL * Nm * N - 1ULL));
	chi2 = (0.5 * chi2 * chi2) / static_cast<double>(N);
	const double ep = ((1. / static_cast<double>(N)) * sum<double>(eps));
	double epi = chi1;
	const double inMean = mean<double>(v);
	double epP, epPi;
	if (computeBayesianP)
		epP = ((1. / static_cast<double>(N)) * sum<double>(epsP));
	//array rho = constant(0.f, Nt - stepSize, 1);
	//array rhoi;
	if (complexType == 1 || complexType == 2) {
		const double inMeanI = mean<double>(vvi);
		epi = ((1. / static_cast<double>(N)) * sum<double>(epsi));
		if (computeBayesianP)
			epPi = ((1. / static_cast<double>(N)) * sum<double>(epsPi));
		//rhoi = constant(0.f, Nt - stepSize, 1);
		mexPrintf("Innovation mean for the real part is %f\n", inMean);
		mexPrintf("Innovation mean for the imaginary part is %f\n", inMeanI);
		mexPrintf("Acceptance interval for NES is [%f, %f]\n", chi1, chi2);
		mexPrintf("Time average NES for the real part is %f\n", ep);
		mexPrintf("Time average NES for the imaginary part is %f\n", epi);
		if (computeBayesianP) {
			mexPrintf("Time average Bayesian p-test for the real part is %f\n", epP);
			mexPrintf("Time average Bayesian p-test for the imaginary part is %f\n", epPi);
		}
	}
	else {
		mexPrintf("Innovation mean is %f\n", inMean);
		mexPrintf("Acceptance interval for NES is [%f, %f]\n", chi1, chi2);
		mexPrintf("Time average NES is %f\n", ep);
		if (computeBayesianP)
			mexPrintf("Time average Bayesian p-test is %f\n", epP);
	}
	const double rhop = (1. / sqrt(static_cast<double>(N))) * sum<double>(sum(v(span, seq(0, N - 1)) * v(span, seq(stepSize, end)), 0)) /
		std::sqrt(sum<double>(sum(v(span, seq(0, N - 1)) * v(span, seq(0, N - 1)), 0)) * sum<double>(sum(v(span, seq(stepSize, end)) * v(span, seq(stepSize, end)), 0)));
	//if (complexType == 1 || complexType == 2)
	//	rhoi(kk) = (1.f / sqrt(static_cast<float>(N))) * sum(matmulTN(vvi(span, kk), vvi(span, kk + stepSize))) * pow(sum(matmulTN(vvi(span, kk), vvi(span, kk))) * sum(matmulTN(vvi(span, kk + stepSize), vvi(span, kk + stepSize))), -.5);
	double rhoInt1 = -1.96 / sqrt(static_cast<double>(N));
	double rhoInt2 = 1.96 / sqrt(static_cast<double>(N));
	//const double rhop = mean<double>(rho);
	double rhopi;
	if (complexType == 1 || complexType == 2) {
		//rhopi = mean<double>(rhoi); 
		const double rhopi = (1. / sqrt(static_cast<double>(N))) * sum<double>(sum(vvi(span, seq(0, N - 1)) * vvi(span, seq(stepSize, end)), 0)) /
			std::sqrt(sum<double>(sum(vvi(span, seq(0, N - 1)) * vvi(span, seq(0, N - 1)), 0)) * sum<double>(sum(vvi(span, seq(stepSize, end)) * vvi(span, seq(stepSize, end)), 0)));
		mexPrintf("Acceptance interval for autocorrelation is [%f, %f]\n", rhoInt1, rhoInt2);
		mexPrintf("Time average autocorrelation for the real part is %f\n", rhop);
		mexPrintf("Time average autocorrelation for the imaginary part is %f\n", rhopi);
	}
	else {
		mexPrintf("Acceptance interval for autocorrelation is [%f, %f]\n", rhoInt1, rhoInt2);
		mexPrintf("Time average autocorrelation is %f\n", rhop);
	}
}

// ETKF and ESTKF computation with regularization
array computeWL(const array& U, const array& SS, const array& HH, const std::vector<array>& R, const array& m0, const array& L, const array& X, const std::vector<array>& S,
	const uint64_t tt, const uint64_t sizeR, const uint64_t hnU, const bool sparseR, const float RR, const uint8_t complexType, const std::vector<array>& Si) {
	if (sparseR)
		if (complexType == 3)
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), matmul(R[tt % sizeR], join(0, real(m0), L, imag(m0), L) - join(0, matmul(S[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) - matmul(Si[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)),
				matmul(Si[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) + matmul(S[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)))));
		else
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), matmul(R[tt % sizeR], join(0, m0, L) - matmul(S[tt % hnU], mean(X, 1))));
	else
		if (complexType == 3)
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
				R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR) * (join(0, real(m0), L, imag(m0), L) - join(0, matmul(S[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) - matmul(Si[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)),
					matmul(Si[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) + matmul(S[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)))));
		else
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR) * (join(0, m0, L) - matmul(S[tt % hnU], mean(X, 1))));
}

// ETKF and ESTKF computation without regularization
array computeW(const array& U, const array& SS, const array& HH, const std::vector<array>& R, const array& m0, const array& X, const std::vector<array>& S,
	const uint64_t tt, const uint64_t sizeR, const uint64_t hnU, const bool sparseR, const float RR, const uint8_t complexType, const std::vector<array>& Si) {
	if (sparseR)
		if (complexType == 3)
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), matmul(R[tt % sizeR], join(0, real(m0), imag(m0)) - join(0, matmul(S[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) - matmul(Si[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)),
				matmul(Si[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) + matmul(S[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)))));
		else
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), matmul(R[tt % sizeR], m0 - matmul(S[tt % hnU], mean(X, 1))));
	else
		if (complexType == 3)
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), R[tt % sizeR] * (join(0, real(m0), imag(m0)) - join(0, matmul(S[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) - matmul(Si[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)),
				matmul(Si[tt % hnU], mean(X(seq(0, X.dims(0) / 2 - 1), span), 1)) + matmul(S[tt % hnU], mean(X(seq(X.dims(0) / 2, end), span), 1)))));
		else
			return matmul(U * tile(1.f / SS.T(), U.dims(0), 1), U.T(), HH.T(), R[tt % sizeR] * (m0 - matmul(S[tt % hnU], mean(X, 1))));
}

void computeAPrioriX(const bool useF, const bool useU, const bool useG, const bool useKineticModel, const uint32_t algorithm, array& xtr, array& xti, const uint64_t imDim,
	const std::vector<array>& F, const std::vector<array>& Fi, const uint8_t complexType, const std::vector<array>& G, const std::vector<array>& Gi, const array& u,
	const bool complexF, const bool complexG, const uint64_t tt, const size_t sizeF, const size_t sizeG, const size_t sizeU) {
	if (useF) {
		if (useKineticModel) {
			xtr(seq(0, F[tt % sizeF].dims(0) - 1)) = matmul(F[tt % sizeF], xtr);
			if (complexType == 1 || complexType == 2)
				xti(seq(0, F[tt % sizeF].dims(0) - 1)) = matmul(F[tt % sizeF], xti);
		}
		else {
			xtr = matmul(F[tt % sizeF], xtr);
			if (complexType == 1 || complexType == 2)
				if (complexF)
					xti = matmul(Fi[tt % sizeF], xti);
				else
					xti = matmul(F[tt % sizeF], xti);
		}
	}
	if (useU) {
		if (useG) {
			xtr += matmul(G[tt % sizeG], real(u(span, tt % sizeU)));
			if (complexType == 1 || complexType == 2)
				if (complexG)
					xti += matmul(Gi[tt % sizeG], imag(u(span, tt % sizeU)));
				else
					xti += matmul(G[tt % sizeG], imag(u(span, tt % sizeU)));
		}
		else {
			if (complexType == 3)
				xtr += join(0, real(u(span, tt % sizeU)), imag(u(span, tt % sizeU)));
			else {
				xtr += real(u(span, tt % sizeU));
				if (complexType == 1 || complexType == 2)
					xti += imag(u(span, tt % sizeU));
			}
		}
	}
}

// Discrete KF
void DKF(array& xt, std::vector<array>& S, std::vector<array>& Si, const array& m0, std::vector<array>& Q, std::vector<array>& Qi, std::vector<array>& R, std::vector<array>& Ri,
	array& Pplus, array& Pplusi, const uint64_t Nt, const uint64_t hn, uint64_t imDim, const uint64_t Nm, const bool storeData, const bool sparseS, const bool complexS,
	array& Svalues, array& Svaluesi, array& Srow, array& Scol, const size_t* sCol, const float* SS3, const float* SS4, const uint8_t complexType, const bool sparseQ,
	const bool sparseR, const size_t sizeQ, const size_t sizeR, const uint64_t hnU, const size_t sizeF, const bool tvF, std::vector<array>& F, std::vector<array>& Fi, const bool useF,
	const bool complexF, const bool useU, const bool useG, const size_t sizeG, std::vector<array>& G, std::vector<array>& Gi, const bool complexG, array& u, const uint32_t algorithm,
	const bool sparseF, const bool fadingMemory, const float fadingAlpha, const uint64_t covIter, const uint64_t N_lag, const bool useSmoother, const bool approximateKS,
	const bool steadyS, const uint32_t skip, bool steadyKF, const uint64_t window, const uint32_t regularization, const uint32_t prior, const uint32_t nIter,
	const TVdata& TV, const TVdata& TVi, const uint32_t Nx, const uint32_t Ny, const uint32_t Nz, const uint32_t Ndx, const uint32_t Ndy, const uint32_t Ndz, const float gamma,
	const float beta, const float betac, const float huberDelta, const array& weightsHuber, const array& weightsQuad, const array& L, const array& Ly, const size_t lSize, const array& Lvalues,
	const array& Lcol, const array& LL, const array& Li, const uint32_t augType, const bool complexRef, const TGVdata TGV, array& xlt, bool useKineticModel,
	const std::vector<float>& S1, const std::vector<float>& S2, const std::vector<int32_t>& sCols, const std::vector<int32_t>& sRows, const float RR, const uint64_t NmU,
	const bool use3D, const uint64_t ensembleSize, const bool computeConsistency, const uint64_t stepSize, const uint64_t initialSteps, const bool computeBayesianP,
	const array& Pred, const bool regType, const uint32_t cgIter, const double cgThreshold, array& P1, const uint8_t storeCovariance, const bool computeInitialValue,
	std::vector<array>& R2, std::vector<array>& R2i, std::vector<array>& Q2, std::vector<array>& Q2i, const bool forceOrthogonalization, const bool useEnsembleMean) //, array& eps, array& epsi, array& v, array& vvi, float& kerroinQ, float& kerroinR) 
{

	// various initializations
	array HH, HHi, SS, SSi, KG, KGi, xtr, xti, xsr, xsi, xs1, Pminus, Pminusi, xtp, xlt1, P, v, vvi, eps, epsi, epsP, epsPi, PI, PIi;
	std::vector<array> KS, KSi;

	bool steady = false;
	bool sGain = false;

	// Denoising type (for different algorithms)
	int32_t Type = 1;
	if (regType)
		Type = 0;

	uint32_t NDimZ = 1U;
	uint32_t DimZ = Nz;
	// If pseudo-3D
	if (!use3D && Nz > 1U) {
		NDimZ = Nz;
		DimZ = 1U;
		//imDim /= static_cast<uint64_t>(NDimZ);
	}
	uint64_t imDimN = imDim;
	uint64_t imDimU = imDim;

	if (DEBUG) {
		mexPrintf("Qi.size() = %d\n", Qi.size());
		mexPrintf("hnU = %d\n", hnU);
		mexEvalString("pause(.0001);");
	}

	if (useKineticModel) {
		imDimN *= 2ULL;
	}
	uint64_t imDim3 = imDimN;
	if (complexType == 3) {
		imDimU *= 2ULL;
		imDim3 *= 2ULL;
	}
	const bool kineticModel = useKineticModel;
	if (algorithm >= 1)
		useKineticModel = false;

	const size_t sizeU = u.dims(1);
	const uint64_t nMeas = Nt * Nm;

	// Pseudo 3D loop
	for (uint32_t NN = 0; NN < NDimZ; NN++) {

		int64_t oo = 0LL, cc = 0LL;
		int jg = 0;
		uint64_t kk = 0ULL;
		double normi = 0.;
		if (DEBUG) {
			mexPrintf("algorithm = %d\n", algorithm);
			mexEvalString("pause(.0001);");
		}

		// "Normal" KF algorithms (regular, information, one-step)
		if (algorithm < 3) {

			// For consistency tests
			if (computeConsistency) {
				// innovation
				v = constant(0.f, NmU, Nt - initialSteps);
				// For NIS
				eps = constant(0.f, Nt - stepSize - initialSteps);
				if (computeBayesianP)
					epsP = constant(0.f, Nt - stepSize - initialSteps);
				if (complexType == 2 || complexType == 1) {
					vvi = constant(0.f, NmU, Nt - initialSteps);
					epsi = constant(0.f, Nt - stepSize - initialSteps);
					if (computeBayesianP)
						epsPi = constant(0.f, Nt - stepSize - initialSteps);
				}
			}

			// Stores the covariance diagonals for steady state KF (convergence inspection)
			if (steadyKF)
				P = constant(1.f, imDimU, hn - (window - 1ULL));

			if (useSmoother) {
				if (complexType == 0 || complexType == 3)
					xlt1 = constant(0.f, imDimU, N_lag + 1);
				else
					xlt1 = constant(0.f, imDimU, N_lag + 1, c32);
				if (!steadyS) {
					KS.resize(N_lag);
					if (complexType == 2)
						KSi.resize(N_lag);
				}
				else {
					KS.resize(1);
					if (complexType == 2)
						KSi.resize(1);
				}
				if (useF || useG || useU) {
					if (complexType == 0 || complexType == 3)
						xtp = constant(0.f, imDimU, N_lag);
					else
						xtp = constant(0.f, imDimU, N_lag, c32);
				}
				if (DEBUG) {
					mexPrintf("xtp.dims(0) = %d\n", xtp.dims(0));
					mexPrintf("xtp.dims(1) = %d\n", xtp.dims(1));
					mexEvalString("pause(.0001);");
				}
			}



			// Preiterate covariance or compute the steady state gain
			if (covIter > 0 || steadyKF) {
				while (!steady) {
					loadSystemMatrix(storeData, sparseS, kk, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
						sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
					if (useSmoother) {
						Pminus = Pplus;
						if (complexType == 2)
							Pminusi = Pplusi;
					}
					if (algorithm == 2) {
						if (complexType == 2) {
							if (complexS)
								oneStepKF(HH, Si, kk, hnU, Pplusi, sparseR, sizeR, Ri, KGi, useF, F, Fi, sizeF, sparseF, RR, regularization, complexF);
							else
								oneStepKF(HH, S, kk, hnU, Pplusi, sparseR, sizeR, Ri, KGi, useF, F, Fi, sizeF, sparseF, RR, regularization, complexF);
							if (computeConsistency)
								HHi = HH;
						}
						oneStepKF(HH, S, kk, hnU, Pplus, sparseR, sizeR, R, KG, useF, F, Fi, sizeF, sparseF, RR, regularization, useKineticModel, complexF);
					}
					if (useF) {
						computePminus(algorithm, sparseF, Pplus, F[kk % sizeF], kk, sizeF, sizeQ, Q[kk % sizeQ], sparseQ, useKineticModel, NN, KG, S, hnU, useF);
						if (complexType == 2)
							if (complexS)
								if (complexF)
									computePminus(algorithm, sparseF, Pplusi, Fi[kk % sizeF], kk, sizeF, sizeQ, Qi[kk % sizeQ], sparseQ, useKineticModel, NN, KG, Si, hnU, useF);
								else
									computePminus(algorithm, sparseF, Pplusi, F[kk % sizeF], kk, sizeF, sizeQ, Qi[kk % sizeQ], sparseQ, useKineticModel, NN, KG, Si, hnU, useF);
							else
								if (complexF)
									computePminus(algorithm, sparseF, Pplusi, Fi[kk % sizeF], kk, sizeF, sizeQ, Qi[kk % sizeQ], sparseQ, useKineticModel, NN, KG, S, hnU, useF);
								else
									computePminus(algorithm, sparseF, Pplusi, F[kk % sizeF], kk, sizeF, sizeQ, Qi[kk % sizeQ], sparseQ, useKineticModel, NN, KG, S, hnU, useF);
					}
					if (fadingMemory) {
						Pplus *= fadingAlpha;
						if (complexType == 2 && algorithm != 2)
							Pplusi *= fadingAlpha;
					}
					if (algorithm == 0) {
						if (DEBUG) {
							mexPrintf("Q[0].dims(0) = %d\n", Q[0].dims(0));
							mexPrintf("Q[0].dims(1) = %d\n", Q[0].dims(1));
							mexEvalString("pause(.0001);");
						}
						if (sparseQ)
							Pplus = Pplus + Q[kk % sizeQ];
						else
							Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q[kk % sizeQ](span, NN);
						if (complexType == 2)
							if (sparseQ)
								Pplusi = Pplusi + Qi[kk % sizeQ];
							else
								Pplusi(seq(0, end, Pplusi.dims(0) + 1)) = Pplusi(seq(0, end, Pplusi.dims(0) + 1)) + Qi[kk % sizeQ](span, NN);
					}
					else if (algorithm == 1 && !useF) {
						if (sparseQ) {
							Pplus = Q[kk % sizeQ] - transpose(matmul(Q[kk % sizeQ], transpose(matmul(Q[kk % sizeQ], inverse(Pplus + Q[kk % sizeQ], AF_MAT_TRANS)))));
							if (complexType == 2)
								Pplusi = Qi[kk % sizeQ] - transpose(matmul(Qi[kk % sizeQ], transpose(matmul(Qi[kk % sizeQ], inverse(Pplusi + Qi[kk % sizeQ], AF_MAT_TRANS)))));
						}
						else {
							Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q[kk % sizeQ](span, NN);
							Pplus = -batchFunc(transpose(Q[kk % sizeQ](span, NN)), batchFunc(Q[kk % sizeQ](span, NN), inverse(Pplus), batchMul), batchMul);
							Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q[kk % sizeQ](span, NN);
							if (complexType == 2) {
								Pplusi(seq(0, end, Pplusi.dims(0) + 1)) = Pplusi(seq(0, end, Pplusi.dims(0) + 1)) + Qi[kk % sizeQ](span, NN);
								Pplusi = -batchFunc(transpose(Qi[kk % sizeQ](span, NN)), batchFunc(Qi[kk % sizeQ](span, NN), inverse(Pplusi), batchMul), batchMul);
								Pplusi(seq(0, end, Pplusi.dims(0) + 1)) = Pplusi(seq(0, end, Pplusi.dims(0) + 1)) + Qi[kk % sizeQ](span, NN);
							}
						}
					}
					if (algorithm == 1) {
						if (sparseR)
							if (complexType == 3)
								Pplus = Pplus + transpose(matmul3(S, Si, transpose(matmul3(S, Si, dense(R[kk % sizeR]), kk % hnU, true)), kk % hnU, true));
							else
								Pplus = Pplus + transpose(matmul(S[kk % hnU], transpose(matmul(S[kk % hnU], dense(R[kk % sizeR]), AF_MAT_TRANS)), AF_MAT_TRANS));
						else
							if (complexType == 3)
								Pplus = Pplus + transpose(matmul3(S, Si, transpose(matmul3(S, Si, diag(R[kk % sizeR], 0, false), kk % hnU, true)), kk % hnU, true));
							else
								Pplus = Pplus + transpose(matmul(S[kk % hnU], transpose(matmul(S[kk % hnU], diag(R[kk % sizeR], 0, false), AF_MAT_TRANS)), AF_MAT_TRANS));
						if (complexType == 2)
							if (complexS)
								if (sparseR)
									Pplusi = Pplusi + transpose(matmul(Si[kk % hnU], transpose(matmul(Si[kk % hnU], dense(Ri[kk % sizeR]), AF_MAT_TRANS)), AF_MAT_TRANS));
								else
									Pplusi = Pplusi + transpose(matmul(Si[kk % hnU], transpose(matmul(Si[kk % hnU], diag(Ri[kk % sizeR], 0, false), AF_MAT_TRANS)), AF_MAT_TRANS));
							else
								if (sparseR)
									Pplusi = Pplusi + transpose(matmul(S[kk % hnU], transpose(matmul(S[kk % hnU], dense(Ri[kk % sizeR]), AF_MAT_TRANS)), AF_MAT_TRANS));
								else
									Pplusi = Pplusi + transpose(matmul(S[kk % hnU], transpose(matmul(S[kk % hnU], diag(Ri[kk % sizeR], 0, false), AF_MAT_TRANS)), AF_MAT_TRANS));
					}
					//mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
					//mexEvalString("pause(.0001);");
					if (algorithm <= 1) {
						if (complexType == 2) {
							if (complexS)
								computeKG(algorithm, KGi, HH, sparseR, Ri, Pplusi, Si, kk, hnU, sizeR, kk % hnU, RR, regularization, complexType, Si);
							else
								computeKG(algorithm, KGi, HH, sparseR, Ri, Pplusi, S, kk, hnU, sizeR, kk % hnU, RR, regularization, complexType, Si);
							if (computeConsistency)
								HHi = HH;
						}
						computeKG(algorithm, KG, HH, sparseR, R, Pplus, S, kk, hnU, sizeR, kk % hnU, RR, regularization, complexType, Si);
					}
					if (algorithm == 0) {
						if (complexType == 3)
							Pplus -= matmul(KG, join(0, matmul(S[kk % hnU], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) - matmul(Si[kk % hnU], Pplus(seq(Pplus.dims(0) / 2, end), span)),
								matmul(Si[kk % hnU], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) + matmul(S[kk % hnU], Pplus(seq(Pplus.dims(0) / 2, end), span))));
						else
							Pplus -= matmul(KG, matmul(S[kk % hnU], Pplus));
						if (complexType == 2) {
							if (complexS)
								Pplusi -= matmul(KGi, matmul(Si[kk % hnU], Pplusi));
							else
								Pplusi -= matmul(KGi, matmul(S[kk % hnU], Pplusi));
						}
					}
					if (algorithm == 0) {
						Pplus = (Pplus + transpose(Pplus)) / 2;
						if (complexType == 2)
							Pplusi = (Pplusi + transpose(Pplusi)) / 2;
					}
					if (computeInitialValue) {
						if (complexType == 3)
							xtr = join(0, real(xt(span, 0, NN)), imag(xt(span, 0, NN)));
						else
							xtr = real(xt(span, 0, NN));
						if (complexType == 1 || complexType == 2)
							xti = imag(xt(span, 0, NN));
						computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, kk, sizeF, sizeG, sizeU);
						if (complexType == 3)
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, kk, S[kk % hnU], xtr, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, kk, S[kk % hnU], xtr, Ly, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, kk, S[kk % hnU], xtr, Ly, Si, complexType, hnU);
						xtr += matmul(KG, SS);
						if (complexType == 0)
							xt(span, 0, NN) = xtr;
						else if (complexType == 1 || complexType == 2) {
							if (complexS)
								if (complexRef && augType > 0 && regularization == 1)
									computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, kk, Si[kk % hnU], xti, Li, Si, complexType, hnU);
								else
									computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, kk, Si[kk % hnU], xti, Ly, Si, complexType, hnU);
							else
								if (complexRef && augType > 0 && regularization == 1)
									computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, kk, S[kk % hnU], xti, Li, Si, complexType, hnU);
								else
									computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, kk, S[kk % hnU], xti, Ly, Si, complexType, hnU);
							if (complexType == 2)
								xti += matmul(KGi, SS);
							else
								xti += matmul(KG, SS);
							xt(span, 0, NN) = complex(xtr, xti);
						}
						else if (complexType == 3)
							if (useKineticModel)
								xt(span, 0, NN) = complex(join(0, xtr(seq(0, imDim - 1)), xtr(seq(imDimU, imDimU + imDim - 1))), join(0, xtr(seq(imDim, imDimU - 1)), xtr(seq(imDimU + imDim, end))));
							else
								xt(span, 0, NN) = complex(xtr(seq(0, xtr.dims(0) / 2 - 1)), xtr(seq(xtr.dims(0) / 2, end)));
					}
					//mexPrintf("KG.summa = %f\n", af::sum<float>(flat(KG)));
					//mexPrintf("kk = %d\n", kk);
					//mexEvalString("pause(.0001);");
					if (steadyKF) {
						//normi = norm(diag(Pplus, 0, true), AF_NORM_EUCLID) / norm(P(span, kk % hn), AF_NORM_EUCLID);
						normi = norm(diag(Pplus, 0, true) - P(span, kk % hn), AF_NORM_EUCLID);
						P(span, kk % hn) = diag(Pplus, 0, true);
						if (DEBUG) {
							if (kk >= hn) {
								mexPrintf("normi = %f\n", normi);
								mexEvalString("pause(.0001);");
							}
						}
					}
					kk++;
					//if ((!steadyKF && kk >= covIter) || (steadyKF && std::fabs(normi - 1.) < 0.005 && kk > hn) || (steadyKF && kk >= 5000))
					//if ((!steadyKF && kk >= covIter) || (steadyKF && std::fabs(normi - 1.) < 0.005 && kk > hn) || (steadyKF && kk >= Nt))
					if ((!steadyKF && kk >= covIter) || (steadyKF && normi < 0.001 && kk > hn) || (steadyKF && kk >= Nt - (window - 1ULL)))
						steady = true;
				}
				if (kk >= Nt - (window - 1ULL) && normi > 0.001) {
					steadyKF = false;
					mexPrintf("Steady state was not reached. Resuming with regular KF.");
				}
				mexEvalString("pause(.0001);");
			}

			for (uint64_t tt = 0ULL; tt < Nt - (window - 1ULL); tt++) {

				jg++;
				loadSystemMatrix(storeData, sparseS, tt, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
					sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
				//mexPrintf("Q[0].dims(0) = %d\n", Q[0].dims(0));
				//mexPrintf("Q[0].dims(1) = %d\n", Q[0].dims(1));
				//mexEvalString("pause(.0001);");

				if (complexType == 3)
					xtr = join(0, real(xt(span, oo, NN)), imag(xt(span, oo, NN)));
				else
					xtr = real(xt(span, oo, NN));
				if (complexType == 1 || complexType == 2)
					xti = imag(xt(span, oo, NN));
				if (DEBUG) {
					mexPrintf("xtr.dims(0) = %d\n", xtr.dims(0));
					mexPrintf("xtr.dims(1) = %d\n", xtr.dims(1));
					mexEvalString("pause(.0001);");
				}
				if (useSmoother && !steadyKF && (!steadyS || (steadyS && tt == N_lag - 1ULL) || (steadyS && tt >= N_lag - 1 && jg == skip) || (steadyS && tt == (Nt - 1)))) {
					//if (useKineticModel)
					//	Pminus = Pplus(seq(0, imDimU - 1), seq(0, imDimU - 1));
					//else
					Pminus = Pplus;
					if (complexType == 2)
						//if (useKineticModel)
						//	Pminusi = Pplusi(seq(0, imDimU - 1), seq(0, imDimU - 1));
						//else
						Pminusi = Pplusi;
				}
				if (!steadyKF) {
					if (algorithm == 2) {
						if (DEBUG) {
							mexPrintf("R[0].dims(0) = %d\n", R[0].dims(0));
							mexPrintf("R[0].dims(1) = %d\n", R[0].dims(1));
							mexEvalString("pause(.0001);");
						}
						if (complexType == 2) {
							if (complexS)
								oneStepKF(HH, Si, tt, hnU, Pplusi, sparseR, sizeR, Ri, KGi, useF, F, Fi, sizeF, sparseF, RR, regularization, complexF);
							else
								oneStepKF(HH, S, tt, hnU, Pplusi, sparseR, sizeR, Ri, KGi, useF, F, Fi, sizeF, sparseF, RR, regularization, complexF);
							if (computeConsistency)
								HHi = HH;
						}
						oneStepKF(HH, S, tt, hnU, Pplus, sparseR, sizeR, R, KG, useF, F, Fi, sizeF, sparseF, RR, regularization, useKineticModel, complexF);
					}
					if (useF) {
						computePminus(algorithm, sparseF, Pplus, F[tt % sizeF], tt, sizeF, sizeQ, Q[tt % sizeQ], sparseQ, useKineticModel, NN, KG, S, hnU, useF);
						if (complexType == 2)
							if (complexS)
								if (complexF)
									computePminus(algorithm, sparseF, Pplusi, Fi[tt % sizeF], tt, sizeF, sizeQ, Qi[tt % sizeQ], sparseQ, useKineticModel, NN, KG, Si, hnU, useF);
								else
									computePminus(algorithm, sparseF, Pplusi, F[tt % sizeF], tt, sizeF, sizeQ, Qi[tt % sizeQ], sparseQ, useKineticModel, NN, KG, Si, hnU, useF);
							else
								if (complexF)
									computePminus(algorithm, sparseF, Pplusi, Fi[tt % sizeF], tt, sizeF, sizeQ, Qi[tt % sizeQ], sparseQ, useKineticModel, NN, KG, S, hnU, useF);
								else
									computePminus(algorithm, sparseF, Pplusi, F[tt % sizeF], tt, sizeF, sizeQ, Qi[tt % sizeQ], sparseQ, useKineticModel, NN, KG, S, hnU, useF);
					}
					if (fadingMemory) {
						Pplus *= fadingAlpha;
						if (complexType == 2 && algorithm != 2)
							Pplusi *= fadingAlpha;
					}
					if (algorithm == 0) {
						if (DEBUG) {
							mexPrintf("Q[0].dims(0) = %d\n", Q[0].dims(0));
							mexPrintf("Q[0].dims(1) = %d\n", Q[0].dims(1));
							mexEvalString("pause(.0001);");
						}
						if (sparseQ)
							Pplus = Pplus + Q[tt % sizeQ];
						else
							Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q[tt % sizeQ](span, NN);
						if (complexType == 2)
							if (sparseQ)
								Pplusi = Pplusi + Qi[tt % sizeQ];
							else
								Pplusi(seq(0, end, Pplusi.dims(0) + 1)) = Pplusi(seq(0, end, Pplusi.dims(0) + 1)) + Qi[tt % sizeQ](span, NN);
					}
					else if (algorithm == 1 && !useF) {
						if (DEBUG) {
							mexPrintf("Q[0].dims(0) = %d\n", Q[0].dims(0));
							mexPrintf("Q[0].dims(1) = %d\n", Q[0].dims(1));
							mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
							mexEvalString("pause(.0001);");
						}
						if (sparseQ) {
							Pplus = Q[tt % sizeQ] - transpose(matmul(Q[tt % sizeQ], transpose(matmul(Q[tt % sizeQ], inverse(Pplus + Q[tt % sizeQ], AF_MAT_TRANS)))));
							if (complexType == 2)
								Pplusi = Qi[tt % sizeQ] - transpose(matmul(Qi[tt % sizeQ], transpose(matmul(Qi[tt % sizeQ], inverse(Pplusi + Qi[tt % sizeQ], AF_MAT_TRANS)))));
						}
						else {
							Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q[tt % sizeQ](span, NN);
							//mexPrintf("Q[tt sizeQ](span, NN).summa = %f\n", af::sum<float>(flat(Q[tt % sizeQ](span, NN))));
							//mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
							//mexEvalString("pause(.0001);");
							Pplus = -batchFunc(transpose(Q[tt % sizeQ](span, NN)), batchFunc(Q[tt % sizeQ](span, NN), inverse(Pplus), batchMul), batchMul);
							//mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
							//mexEvalString("pause(.0001);");
							Pplus(seq(0, end, Pplus.dims(0) + 1)) = Pplus(seq(0, end, Pplus.dims(0) + 1)) + Q[tt % sizeQ](span, NN);
							if (complexType == 2) {
								Pplusi(seq(0, end, Pplusi.dims(0) + 1)) = Pplusi(seq(0, end, Pplusi.dims(0) + 1)) + Qi[tt % sizeQ](span, NN);
								Pplusi = -batchFunc(transpose(Qi[tt % sizeQ](span, NN)), batchFunc(Qi[tt % sizeQ](span, NN), inverse(Pplusi), batchMul), batchMul);
								Pplusi(seq(0, end, Pplusi.dims(0) + 1)) = Pplusi(seq(0, end, Pplusi.dims(0) + 1)) + Qi[tt % sizeQ](span, NN);
							}
						}
						//mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
						//mexEvalString("pause(.0001);");
					}
				}
				if (useSmoother && (!steadyS || (steadyS && tt == N_lag - 1ULL) || (steadyS && tt >= N_lag - 1 && jg == skip) || (steadyS && tt == (Nt - 1)))) {
					uint64_t dimS = oo % N_lag;
					if (steadyS)
						dimS = 0;
					if (!steadyKF || (steadyKF && !sGain)) {
						if (approximateKS && algorithm == 0) {
							if (useF) {
								if (complexType < 4) {
									if (useKineticModel) {
										Pminus(seq(0, Pminus.dims(0) / 2 - 1), span) = matmul(F[tt % sizeF], Pminus);
										KS[dimS] = transpose(batchFunc(Pminus, diag(Pplus, 0, true), batchDiv));
										KS[dimS] = KS[dimS](seq(0, KS[dimS].dims(0) / 2 - 1), seq(0, KS[dimS].dims(1) / 2 - 1));
									}
									else
										KS[dimS] = transpose(batchFunc(matmul(F[tt % sizeF], Pminus), diag(Pplus, 0, true), batchDiv));
									if (complexType == 2)
										if (useKineticModel) {
											Pminusi(seq(0, Pminusi.dims(0) / 2 - 1), span) = matmul(F[tt % sizeF], Pminusi);
											KSi[dimS] = transpose(batchFunc(Pminusi, diag(Pplusi, 0, true), batchDiv));
											KSi[dimS] = KSi[dimS](seq(0, KSi[dimS].dims(0) / 2 - 1), seq(0, KSi[dimS].dims(1) / 2 - 1));
										}
										else
											KSi[dimS] = transpose(batchFunc(matmul(Fi[tt % sizeF], Pminusi), diag(Pplusi, 0, true), batchDiv));
								}
								else
									KS[dimS] = batchFunc(transpose(matmul(F[tt % sizeF], Pminus), true), diag(Pplus, 0, true), batchDiv);
							}
							else {
								KS[dimS] = batchFunc(Pminus, diag(Pplus, 0, true), batchDiv);
								if (complexType == 2)
									KSi[dimS] = batchFunc(Pminusi, diag(Pplusi, 0, true), batchDiv);
							}
						}
						else {
							if (algorithm == 0) {
								if (useF) {
									if (sparseF) {
										if (complexType < 4) {
											if (useKineticModel) {
												Pminus(seq(0, Pminus.dims(0) / 2 - 1), span) = matmul(F[tt % sizeF], Pminus);
												KS[dimS] = transpose(solve(Pplus, Pminus));
												KS[dimS] = KS[dimS](seq(0, KS[dimS].dims(0) / 2 - 1), seq(0, KS[dimS].dims(1) / 2 - 1));
											}
											else
												KS[dimS] = transpose(solve(Pplus, matmul(F[tt % sizeF], Pminus)));
											if (complexType == 2) {
												if (complexF)
													KSi[dimS] = transpose(solve(Pplusi, matmul(Fi[tt % sizeF], Pminusi)));
												else
													if (useKineticModel) {
														Pminusi(seq(0, Pminusi.dims(0) / 2 - 1), span) = matmul(F[tt % sizeF], Pminusi);
														KSi[dimS] = transpose(solve(Pplusi, Pminusi));
														KSi[dimS] = KSi[dimS](seq(0, KSi[dimS].dims(0) / 2 - 1), seq(0, KSi[dimS].dims(1) / 2 - 1));
													}
													else
														KSi[dimS] = transpose(solve(Pplusi, matmul(F[tt % sizeF], Pminusi)));
											}
										}
										else
											KS[dimS] = transpose(solve(Pplus, matmul(F[tt % sizeF], Pminus)), true);
									}
									else {
										if (complexType < 4) {
											KS[dimS] = matmulNT(Pminus, solve(Pplus, F[tt % sizeF]));
											if (complexType == 2)
												if (complexF)
													KSi[dimS] = matmulNT(Pminusi, solve(Pplusi, Fi[tt % sizeF]));
												else
													KSi[dimS] = matmulNT(Pminusi, solve(Pplusi, F[tt % sizeF]));
										}
										else
											KS[dimS] = matmul(Pminus, transpose(solve(Pplus, F[tt % sizeF]), true));
									}
								}
								else {
									KS[dimS] = transpose(solve(Pplus, Pminus));
									if (complexType == 2)
										KSi[dimS] = transpose(solve(Pplusi, Pminusi));
								}
							}
							else if (algorithm == 1) {
								if (useF) {
									if (sparseF) {
										if (complexType < 4) {
											//if (useKineticModel)
											//	KS[dimS] = matmul(PI(seq(0, imDimU - 1), seq(0, imDimU - 1)), matmul(F[tt % sizeF], Pminus, AF_MAT_TRANS));
											//else
											KS[dimS] = matmul(PI, matmul(F[tt % sizeF], Pminus, AF_MAT_TRANS));
											if (complexType == 2) {
												if (complexF)
													KSi[dimS] = matmul(PIi, matmul(Fi[tt % sizeF], Pminusi, AF_MAT_TRANS));
												else
													//if (useKineticModel)
													//	KSi[dimS] = matmul(PIi(seq(0, imDimU - 1), seq(0, imDimU - 1)), matmul(F[tt % sizeF], Pminusi, AF_MAT_TRANS));
													//else
													KSi[dimS] = matmul(PIi, matmul(F[tt % sizeF], Pminusi, AF_MAT_TRANS));
											}
										}
										else
											KS[dimS] = matmul(PI, matmul(F[tt % sizeF], Pminus, AF_MAT_CTRANS));
									}
									else {
										if (complexType < 4) {
											KS[dimS] = matmul(PI, F[tt % sizeF].T(), Pminus);
											if (complexType == 2)
												if (complexF)
													KSi[dimS] = matmul(PIi, Fi[tt % sizeF].T(), Pminusi);
												else
													KSi[dimS] = matmul(PIi, F[tt % sizeF].T(), Pminusi);
										}
										else
											KS[dimS] = matmul(PI, matmul(F[tt % sizeF], Pminus, AF_MAT_CTRANS));
									}
								}
								else {
									KS[dimS] = matmul(PI, Pminus);
									if (complexType == 2)
										KSi[dimS] = matmul(PIi, Pminusi);
								}
							}
						}
						sGain = true;
						if (DEBUG) {
							mexPrintf("KS[dimS].dims(0) = %d\n", KS[dimS].dims(0));
							mexPrintf("KS[dimS].dims(1) = %d\n", KS[dimS].dims(1));
							mexPrintf("KS[dimS].summa = %f\n", af::sum<float>(flat(KS[dimS])));
							mexEvalString("pause(.0001);");
						}
					}
				}
				if (!steadyKF) {
					if (algorithm == 1) {
						if (sparseR)
							if (complexType == 3)
								Pplus = Pplus + transpose(matmul3(S, Si, transpose(matmul3(S, Si, dense(R[tt % sizeR]), tt % hnU, true)), tt % hnU, true));
							else
								if (sparseS)
									Pplus = Pplus + matmul(S[tt % hnU], matmul(R[tt % sizeR], dense(S[tt % hnU])), AF_MAT_TRANS);
								else
									Pplus = Pplus + matmul(S[tt % hnU], matmul(R[tt % sizeR], S[tt % hnU]), AF_MAT_TRANS);
						else
							if (complexType == 3)
								Pplus = Pplus + transpose(matmul3(S, Si, transpose(matmul3(S, Si, diag(R[tt % sizeR], 0, false), tt % hnU, true)), tt % hnU, true));
							else
								if (sparseS)
									if (regularization == 1)
										Pplus = Pplus + matmul(S[tt % hnU], tile(join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, S[tt % hnU].dims(1)) * dense(S[tt % hnU]), AF_MAT_TRANS);
									else
										Pplus = Pplus + matmul(S[tt % hnU], dense(S[tt % hnU]) * tile(R[tt % sizeR], 1, S[tt % hnU].dims(1)), AF_MAT_TRANS);
								else
									if (regularization == 1)
										Pplus = Pplus + matmul(S[tt % hnU], tile(join(0, R[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, S[tt % hnU].dims(1)) * S[tt % hnU], AF_MAT_TRANS);
									else
										Pplus = Pplus + matmul(S[tt % hnU], S[tt % hnU] * tile(R[tt % sizeR], 1, S[tt % hnU].dims(1)), AF_MAT_TRANS);
						if (complexType == 2) {
							if (complexS)
								if (sparseR)
									if (sparseS)
										Pplusi = Pplusi + matmul(Si[tt % hnU], matmul(Ri[tt % sizeR], dense(Si[tt % hnU])), AF_MAT_TRANS);
									else
										Pplusi = Pplusi + matmul(Si[tt % hnU], matmul(Ri[tt % sizeR], (Si[tt % hnU])), AF_MAT_TRANS);
								else
									if (sparseS)
										if (regularization == 1)
											Pplusi = Pplusi + matmul(Si[tt % hnU], tile(join(0, Ri[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, Si[tt % hnU].dims(1)) * dense(Si[tt % hnU]), AF_MAT_TRANS);
										else
											Pplusi = Pplusi + matmul(Si[tt % hnU], tile(Ri[tt % sizeR], 1, Si[tt % hnU].dims(1)) * dense(Si[tt % hnU]), AF_MAT_TRANS);
									else
										if (regularization == 1)
											Pplusi = Pplusi + matmul(Si[tt % hnU], tile(join(0, Ri[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, Si[tt % hnU].dims(1)) * Si[tt % hnU], AF_MAT_TRANS);
										else
											Pplusi = Pplusi + matmul(Si[tt % hnU], tile(Ri[tt % sizeR], 1, Si[tt % hnU].dims(1)) * Si[tt % hnU], AF_MAT_TRANS);
							else
								if (sparseR)
									if (sparseS)
										Pplusi = Pplusi + matmul(S[tt % hnU], matmul(Ri[tt % sizeR], dense(S[tt % hnU])), AF_MAT_TRANS);
									else
										Pplusi = Pplusi + matmul(S[tt % hnU], matmul(Ri[tt % sizeR], (S[tt % hnU])), AF_MAT_TRANS);
								else
									if (sparseS)
										if (regularization == 1)
											Pplusi = Pplusi + matmul(S[tt % hnU], tile(join(0, Ri[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, S[tt % hnU].dims(1)) * dense(S[tt % hnU]), AF_MAT_TRANS);
										else
											Pplusi = Pplusi + matmul(S[tt % hnU], tile(Ri[tt % sizeR], 1, S[tt % hnU].dims(1)) * dense(S[tt % hnU]), AF_MAT_TRANS);
									else
										if (regularization == 1)
											Pplusi = Pplusi + matmul(S[tt % hnU], tile(join(0, Ri[tt % sizeR], constant(1.f, Pplus.dims(0), 1) * RR), 1, S[tt % hnU].dims(1)) * S[tt % hnU], AF_MAT_TRANS);
										else
											Pplusi = Pplusi + matmul(S[tt % hnU], tile(Ri[tt % sizeR], 1, S[tt % hnU].dims(1)) * S[tt % hnU], AF_MAT_TRANS);
							PIi = inverse(Pplusi);
						}
						//mexPrintf("Pplus.summa2 = %f\n", af::sum<float>(flat(Pplus)));
						//mexPrintf("invsere(Pplus).summa2 = %f\n", af::sum<float>(flat(inverse(Pplus + 1e-8f))));
						//mexEvalString("pause(.0001);");
						//Pplus = diag(1.f / diag(Pplus), 0, false);
						PI = inverse(Pplus);
					}
					//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
					//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
					if (DEBUG) {
						mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
						mexEvalString("pause(.0001);");
					}
					if (complexType == 3)
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
					else
						computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
					//}
					if (DEBUG) {
						mexPrintf("SS.dims(0) = %d\n", SS.dims(0));
						mexPrintf("SS.dims(1) = %d\n", SS.dims(1));
						mexPrintf("SS.summa = %f\n", af::sum<float>(flat(SS)));
						mexEvalString("pause(.0001);");
					}
					if (algorithm <= 1) {
						if (complexType == 2) {
							if (algorithm == 1) {
								if (complexS)
									if (complexRef && augType > 0 && regularization == 1)
										computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
									else
										computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
								else
									if (complexRef && augType > 0 && regularization == 1)
										computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
									else
										computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
							}
							if (complexS)
								computeKG(algorithm, KGi, HH, sparseR, Ri, Pplusi, Si, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, PIi, SSi);
							else
								computeKG(algorithm, KGi, HH, sparseR, Ri, Pplusi, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, PIi, SSi);
							if (computeConsistency)
								HHi = HH;
						}
						computeKG(algorithm, KG, HH, sparseR, R, Pplus, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, PI);
					}
					af::eval(KG);
					if (DEBUG) {
						mexPrintf("KG.summa = %f\n", af::sum<float>(flat(KG)));
						mexEvalString("pause(.0001);");
					}
				}
				//if (algorithm == 2) {
				//	computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
				//	computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
				//}
				//else {
				computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
				if (DEBUG) {
					mexPrintf("xtr.dims(0) = %d\n", xtr.dims(0));
					mexPrintf("xtr.dims(1) = %d\n", xtr.dims(1));
					mexPrintf("xtr.summa = %f\n", af::sum<float>(flat(xtr)));
					mexEvalString("pause(.0001);");
				}
				if (computeConsistency && tt >= initialSteps) {
					storeConsistency(tt, Nt, stepSize, eps, SS, HH, cc, v, sparseR, regularization, Pplus, epsP, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, false);
				}
				if (useSmoother) {
					if (useF || useG || useU)
						if (complexType == 0 || complexType == 3)
							xtp(span, oo % N_lag) = xtr(seq(0, imDimU - 1));
						else
							xtp(span, oo % N_lag) = complex(xtr(seq(0, imDimU - 1)), xti(seq(0, imDimU - 1)));
				}
				if (DEBUG) {
					mexPrintf("KG.summa = %f\n", af::sum<float>(flat(KG)));
					mexEvalString("pause(.0001);");
				}
				xtr += matmul(KG, SS);
				if (algorithm == 0 && !steadyKF) {
					if (complexType == 3)
						Pplus -= matmul(KG, join(0, matmul(S[tt % hnU], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], Pplus(seq(Pplus.dims(0) / 2, end), span)),
							matmul(Si[tt % hnU], Pplus(seq(0, Pplus.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], Pplus(seq(Pplus.dims(0) / 2, end), span))));
					else
						Pplus -= matmul(KG, matmul(S[tt % hnU], Pplus));
					mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
					mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
					mexEvalString("pause(.0001);");
				}
				if (computeConsistency && tt >= initialSteps && computeBayesianP) {
					computeInnovationCov(HH, sparseR, R, Pplus, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si);
					if (complexType == 3)
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
					else
						computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
					storeConsistency(tt, Nt, stepSize, eps, SS, HH, cc, v, sparseR, regularization, Pplus, epsP, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, true);
				}
				if (complexType == 0)
					xt(span, oo + 1, NN) = xtr;
				else if (complexType == 1 || complexType == 2) {
					if (complexS)
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
					else
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
					if (computeConsistency && tt >= initialSteps) {
						if (complexType == 2)
							storeConsistency(tt, Nt, stepSize, epsi, SS, HHi, cc, vvi, sparseR, regularization, Pplusi, epsPi, computeBayesianP, Ri[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, false);
						else
							storeConsistency(tt, Nt, stepSize, epsi, SS, HH, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, false);
					}
					if (complexType == 2)
						xti += matmul(KGi, SS);
					else
						xti += matmul(KG, SS);
					xt(span, oo + 1, NN) = complex(xtr, xti);
					if (algorithm == 0 && !steadyKF && complexType == 2) {
						if (complexS)
							Pplusi -= matmul(KGi, matmul(Si[tt % hnU], Pplusi));
						else
							Pplusi -= matmul(KGi, matmul(S[tt % hnU], Pplusi));
					}
					if (computeConsistency && tt >= initialSteps && computeBayesianP) {
						if (complexType == 2)
							computeInnovationCov(HHi, sparseR, Ri, Pplusi, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, complexS);
						else
							computeInnovationCov(HH, sparseR, R, Pplus, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si);
						if (complexS)
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
						else
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
						if (complexType == 2)
							if (complexS)
								storeConsistency(tt, Nt, stepSize, epsi, SS, HHi, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, Si[tt % hnU], complexType, Si, hnU, true);
							else
								storeConsistency(tt, Nt, stepSize, epsi, SS, HHi, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, true);
						else
							storeConsistency(tt, Nt, stepSize, epsi, SS, HH, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, true);
					}
				}
				else if (complexType == 3)
					if (useKineticModel)
						xt(span, oo + 1, NN) = complex(join(0, xtr(seq(0, imDim - 1)), xtr(seq(imDimU, imDimU + imDim - 1))), join(0, xtr(seq(imDim, imDimU - 1)), xtr(seq(imDimU + imDim, end))));
					else
						xt(span, oo + 1, NN) = complex(xtr(seq(0, xtr.dims(0) / 2 - 1)), xtr(seq(xtr.dims(0) / 2, end)));
				if (computeConsistency && tt >= initialSteps)
					cc++;
				if (complexType == 4) {
					if (useF)
						Pplus = matmulNT(matmul(F[tt % sizeF], Pplus), F[tt % sizeF]);
					if (sparseQ)
						Pplus = Pplus + Q[tt % sizeQ];
					else
						Pplus(seq(0, end, imDim + 1)) = Pplus(seq(0, end, imDim + 1)) + Q[tt % sizeQ](span, NN);
					if (fadingMemory) {
						real(Pplus) = real(Pplus) * fadingAlpha;
						imag(Pplus) = imag(Pplus) * fadingAlpha;
					}
					xtr = xt(span, tt, NN);
					if (useF)
						xtr = matmul(F[tt % sizeF], xtr);
					if (useU) {
						if (useG)
							xtr += matmul(G[tt % sizeG], u(span, tt % sizeU));
						else
							xtr += u(span, tt % sizeU);
					}
					if (useSmoother) {
						if (useF || useG || useU)
							xtp(span, oo % N_lag) = xtr;
					}
					HH = matmul(matmul(S[tt % hnU], Pplus), transpose(S[tt % hnU], true));
					if (sparseR)
						HH = HH + R[tt % sizeR];
					else
						HH(seq(0, end, HH.dims(0) + 1)) = HH(seq(0, end, HH.dims(0) + 1)) + R[tt % sizeR];
					KG = transpose(solve(transpose(HH, true), matmul(S[tt % hnU], transpose(Pplus, true))), true);
					if (regularization == 1)
						SS = join(0, m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))), Ly) - matmul(S[tt % hnU], xtr);
					else
						SS = (m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))) - matmul(S[tt % hnU], xtr);
					xtr += matmul(KG, SS);
					Pplus -= matmul(KG, matmul(S[tt % hnU], Pplus));
					xt(span, oo + 1, NN) = xtr;
				}

				if (DEBUG) {
					mexPrintf("xtr.summa = %f\n", af::sum<float>(flat(xtr)));
					mexPrintf("xtr.min = %f\n", af::min<float>(abs(flat(xtr))));
					mexEvalString("pause(.0001);");
				}
				if (regularization > 2) {
					computeDenoising(xt, imDim, oo, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
						huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, Type);
					if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
						mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
						break;
					}
				}

				if (useSmoother && ((oo == N_lag - 1) || (oo >= N_lag - 1 && jg == skip) || oo == (Nt - 1))) {
					jg = 0;
					int ww = 0;
					int ll = oo % N_lag;
					int jj = oo % N_lag;
					if (steadyS)
						ll = 0;
					if (complexType == 3)
						xlt1(span, end) = join(0, real(xt(seq(0, imDim - 1), oo + 1, NN)), imag(xt(seq(0, imDim - 1), oo + 1, NN)));
					else
						xlt1(span, end) = xt(seq(0, imDim - 1), oo + 1, NN);
					if (DEBUG) {
						mexPrintf("N_lag = %d\n", N_lag);
						mexPrintf("oo = %d\n", oo);
						mexPrintf("xlt1.dims(0) = %d\n", xlt1.dims(0));
						mexPrintf("xlt1.dims(1) = %d\n", xlt1.dims(1));
						mexEvalString("pause(.0001);");
					}
					for (int64_t to = N_lag - 1; to >= 0; to--) {
						//if (to == N_lag - 1) {
						//	//xtp_old = xtp(span, jj);
						//}
						//else {
						//	//xtp_old = xtp(span, jj - ww);
						//}
						if (useF || useG || useU) {
							if (complexType == 0)
								xlt1(span, to) = xt(seq(0, imDim - 1), oo - ww) + matmul(KS[ll], (xlt1(span, to + 1) - xtp(span, jj)));
							else if (complexType == 1)
								xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KS[ll], (real(xlt1(span, to + 1)) - real(xtp(span, jj)))), imag(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KS[ll], (imag(xlt1(span, to + 1)) - imag(xtp(span, jj)))));
							else if (complexType == 2)
								xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KS[ll], (real(xlt1(span, to + 1)) - real(xtp(span, jj)))), imag(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KSi[ll], (imag(xlt1(span, to + 1)) - imag(xtp(span, jj)))));
							else if (complexType == 3)
								xlt1(span, to) = join(0, real(xt(seq(0, imDim - 1), oo - ww, NN)), imag(xt(seq(0, imDim - 1), oo - ww, NN))) + matmul(KS[ll], xlt1(span, to + 1) - xtp(span, jj));
						}
						else {
							if (complexType == 0)
								xlt1(span, to) = xt(seq(0, imDim - 1), oo - ww) + matmul(KS[ll], (xlt1(span, to + 1) - xt(seq(0, imDim - 1), oo - ww, NN)));
							else if (complexType == 1)
								xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KS[ll], (real(xlt1(span, to + 1)) - real(xt(seq(0, imDim - 1), oo - ww, NN)))), imag(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KS[ll], (imag(xlt1(span, to + 1)) - imag(xt(seq(0, imDim - 1), oo - ww, NN)))));
							else if (complexType == 2)
								xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KS[ll], (real(xlt1(span, to + 1)) - real(xt(seq(0, imDim - 1), oo - ww, NN)))), imag(xt(seq(0, imDim - 1), oo - ww, NN)) + matmul(KSi[ll], (imag(xlt1(span, to + 1)) - imag(xt(seq(0, imDim - 1), oo - ww, NN)))));
							else if (complexType == 3)
								xlt1(span, to) = join(0, real(xt(seq(0, imDim - 1), oo - ww, NN)), imag(xt(seq(0, imDim - 1), oo - ww, NN))) + matmul(KS[ll], xlt1(span, to + 1) - join(0, real(xt(seq(0, imDim - 1), oo - ww, NN)), imag(xt(seq(0, imDim - 1), oo - ww, NN))));
						}
						if (DEBUG) {
							mexPrintf("xlt1(span, to + 1).summa = %f\n", af::sum<float>(flat(real(xlt1(span, to + 1)))));
							mexPrintf("xlt1(span, to + 1).summa = %f\n", af::sum<float>(flat(real(xt(seq(0, imDim - 1), oo - ww + 1, NN)))));
							mexPrintf("ww = %d\n", ww);
						}
						ww++;
						jj--;
						ll--;
						if (jj < 0)
							jj = N_lag - 1;
						if (ll < 0 && !steadyS)
							ll = N_lag - 1;
						else if (steadyS)
							ll = 0;
					}
					if (complexType == 3)
						xlt(span, seq(oo + 1LL - (N_lag), oo), NN) = complex(xlt1(seq(0, xlt1.dims(0) / 2 - 1), seq(0, end - 1LL)), xlt1(seq(xlt1.dims(0) / 2, end), seq(0, end - 1LL)));
					else
						xlt(span, seq(oo + 1LL - (N_lag), oo), NN) = xlt1(span, seq(0, N_lag - 1LL));
				}
				oo++;

				if (algorithm == 0 && !steadyKF && oo < Nt) {
					Pplus = (Pplus + transpose(Pplus)) / 2.f;
					if (complexType == 2)
						Pplusi = (Pplusi + transpose(Pplusi)) / 2.f;
				}
				if (storeCovariance == 1) {
					if (complexType == 2)
						if (useKineticModel)
							P1(span, tt) = complex(diag(Pplus(seq(0, imDim - 1), seq(0, imDim - 1)), 0, true), diag(Pplusi(seq(0, imDim - 1), seq(0, imDim - 1)), 0, true));
						else
							P1(span, tt) = complex(diag(Pplus, 0, true), diag(Pplusi, 0, true));
					else if (complexType == 3)
						P1(span, tt) = complex(diag(Pplus(seq(0, imDim - 1), seq(0, imDim - 1)), 0, true), diag(Pplus(seq(imDim, end), seq(imDim, end)), 0, true));
					else
						if (useKineticModel)
							P1(span, tt) = diag(Pplus(seq(0, imDim - 1), seq(0, imDim - 1)), 0, true);
						else
							P1(span, tt) = diag(Pplus, 0, true);
				}

				////Pplus(span, oo) = diag(Pplus, 0, true);
				if (DEBUG) {
					mexPrintf("tt = %d\n", tt);
				}
				//}

			}
			deviceGC();
			if (storeCovariance == 2)
				if (complexType == 2)
					if (useKineticModel)
						P1 = complex(Pplus(seq(0, imDim - 1), seq(0, imDim - 1)), Pplusi(seq(0, imDim - 1), seq(0, imDim - 1)));
					else
						P1 = complex(Pplus, Pplusi);
				else if (complexType == 3)
					P1 = complex(Pplus(seq(0, imDim - 1), seq(0, imDim - 1)), Pplus(seq(imDim, end), seq(imDim, end)));
				else
					if (useKineticModel)
						P1 = Pplus(seq(0, imDim - 1), seq(0, imDim - 1));
					else
						P1 = Pplus;
			if (computeConsistency) {
				computeConsistencyTests(cc, Nt, stepSize, initialSteps, v, Nm, eps, epsP, computeBayesianP, complexType, vvi, epsi, epsPi);
			}
			if (DEBUG) {
				mexPrintf("xt.dims(0) = %d\n", xt.dims(0));
				mexPrintf("xt.dims(1) = %d\n", xt.dims(1));
				mexPrintf("xt.dims(2) = %d\n", xt.dims(2));
			}
			//if (useKineticModel && NDimZ == 1)
			//	xt = xt(seq(0, xt.dims(0) / 2 - 1), span);
		}
		// Ensemble filters
		else if (algorithm >= 3 && algorithm <= 8) {

			array X, Xi, Y, Yi, QQ, Z, RA, SR, U, SS, V, AA, XSk, XSki, PP, Rapu, Rapui;
			array* R1, * R1i;

			std::vector<array> GS, GSi, XS, XSi;

			Type = 1;


			if (useSmoother) {
				GS.resize(N_lag);
				XS.resize(N_lag);
				if (complexType == 1 || complexType == 2) {
					GSi.resize(N_lag);
					XSi.resize(N_lag);
					XSki = tile(imag(xt(span, 0, NN)), 1, ensembleSize);
					XSk = tile(real(xt(span, 0, NN)), 1, ensembleSize);
				}
				else if (complexType == 3)
					XSk = tile(join(0, real(xt(span, 0, NN)), imag(xt(span, 0, NN))), 1, ensembleSize);
				else
					XSk = tile(xt(span, 0, NN), 1, ensembleSize);
			}

			if (algorithm == 8) {
				const float ensemble_s = static_cast<float>(ensembleSize);
				AA = constant((-(1.f / ensemble_s) * (1.f / (1.f / sqrt(ensemble_s) + 1.f))), ensemble_s, ensembleSize - 1, f32);
				AA(seq(0, end, ensembleSize + 1)) = 1.f - (1.f / ensemble_s) * (1.f / (1.f / sqrt(ensemble_s) + 1.f));
				AA(end, span) = -1.f / sqrt(ensemble_s);
				//mexPrintf("AA.dims(0) = %d\n", AA.dims(0));
				//mexPrintf("AA.dims(1) = %d\n", AA.dims(1));
				//mexEvalString("pause(.0001);");
			}
			const float ensembleF = static_cast<float>(ensembleSize - 1);
			const float ensembleD = 1.f / ensembleF;
			const float ensembleD2 = 1.f / static_cast<float>(ensembleSize);
			const float ensembleS = std::sqrt(ensembleF);

			if (!useEnsembleMean) {
				X = tile(real(xt(span, 0, NN)), 1, ensembleSize);
				Xi = tile(imag(xt(span, 0, NN)), 1, ensembleSize);
			}

			for (uint64_t tt = 0ULL; tt < Nt - (window - 1ULL); tt++) {
				jg++;
				loadSystemMatrix(storeData, sparseS, tt, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
					sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
				if (complexType <= 3) {
					if (useEnsembleMean) {
						if (complexType == 0)
							X = xt(span, tt, NN);
						else if (complexType == 3)
							X = join(0, real(xt(span, tt, NN)), imag(xt(span, tt, NN)));
						else {
							X = real(xt(span, tt, NN));
							Xi = imag(xt(span, tt, NN));
						}
					}
					//mexPrintf("X.dims(0) = %d\n", X.dims(0));
					//mexPrintf("X.dims(1) = %d\n", X.dims(1));
					//mexEvalString("pause(.0001);");
					computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, X, Xi, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
					if (sparseQ) {
						if (useEnsembleMean)
							X = tile(X, 1, ensembleSize) + matmul(Q[tt % sizeQ], randn(X.dims(0), ensembleSize));
						else
							X += matmul(Q[tt % sizeQ], randn(X.dims(0), ensembleSize));
						//X = moddims(X, imDimN, ensembleSize);
						if (complexType == 1)
							if (useEnsembleMean)
								Xi = tile(Xi, 1, ensembleSize) + matmul(Q[tt % sizeQ], randn(Xi.dims(0), ensembleSize));
							else
								Xi += matmul(Q[tt % sizeQ], randn(Xi.dims(0), ensembleSize));
						else if (complexType == 2)
							if (useEnsembleMean)
								Xi = tile(Xi, 1, ensembleSize) + matmul(Qi[tt % sizeQ], randn(Xi.dims(0), ensembleSize));
							else
								Xi += matmul(Qi[tt % sizeQ], randn(Xi.dims(0), ensembleSize));
						//if (complexType == 1 || complexType == 2)
						//	Xi = moddims(Xi, imDimN, ensembleSize);
					}
					else {
						if (!useEnsembleMean)
							X += randn(X.dims(0), ensembleSize) * tile(Q[tt % sizeQ](span, NN), 1, ensembleSize);
						else
							X = tile(X, 1, ensembleSize) + randn(X.dims(0), ensembleSize) * tile(Q[tt % sizeQ](span, NN), 1, ensembleSize);
						//mexPrintf("X.dims(0) = %d\n", X.dims(0));
						//mexPrintf("X.dims(1) = %d\n", X.dims(1));
						//mexEvalString("pause(.0001);");
						if (complexType == 1)
							if (!useEnsembleMean)
								Xi += randn(Xi.dims(0), ensembleSize) * tile(Q[tt % sizeQ](span, NN), 1, ensembleSize);
							else
								Xi = tile(Xi, 1, ensembleSize) + randn(Xi.dims(0), ensembleSize) * tile(Q[tt % sizeQ](span, NN), 1, ensembleSize);
						else if (complexType == 2)
							if (!useEnsembleMean)
								Xi += randn(Xi.dims(0), ensembleSize) * tile(Qi[tt % sizeQ](span, NN), 1, ensembleSize);
							else
								Xi = tile(Xi, 1, ensembleSize) + randn(Xi.dims(0), ensembleSize) * tile(Qi[tt % sizeQ](span, NN), 1, ensembleSize);
					}
					//mexPrintf("Xi.dims(0) = %d\n", Xi.dims(0));
					//mexPrintf("Xi.dims(1) = %d\n", Xi.dims(1));
					//mexEvalString("pause(.0001);");
				}
				if (algorithm >= 3 && algorithm <= 6) {
					if (algorithm == 4 || algorithm == 3) {
						R1 = &R[tt % sizeR];
						if (complexType == 2)
							R1i = &Ri[tt % sizeR];
					}
					else {
						if (sparseR) {
							R1 = &R2[tt % sizeR];
							if (complexType == 2)
								R1i = &R2i[tt % sizeR];
						}
						else {
							Rapu = (1.f / R[tt % sizeR]);
							R1 = &Rapu;
							if (complexType == 2) {
								Rapui = (1.f / Ri[tt % sizeR]);
								R1i = &Rapui;
							}
						}
					}
					if (complexType <= 3) {
						if (regularization == 1) {
							if (complexType == 0)
								Y = join(0, m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))), Ly);
							else if (complexType == 3) {
								if (complexRef && augType > 0)
									Y = join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly,
										imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li);
								else
									Y = join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly,
										imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly);
							}
							else {
								Y = join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly);
								if (complexRef && augType > 0)
									Yi = join(0, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li);
								else
									Yi = join(0, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly);
							}
							if (sparseR) {
								Y = tile(Y, 1, ensembleSize) + matmul(R[tt % sizeR], randn(NmU + Ly.dims(0), ensembleSize));
								if (complexType == 1)
									Yi = tile(Yi, 1, ensembleSize) + matmul(R[tt % sizeR], randn(NmU + Ly.dims(0), ensembleSize));
								else if (complexType == 2)
									Yi = tile(Yi, 1, ensembleSize) + matmul(Ri[tt % sizeR], randn(NmU + Ly.dims(0), ensembleSize));
								//if (complexType == 1 || complexType == 2)
								//	Yi = moddims(Yi, Nm, ensembleSize);
							}
							else {
								Y = tile(Y, 1, ensembleSize) + tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, ensembleSize) * randn(Y.dims(0), ensembleSize);
								if (complexType == 1)
									Yi = tile(Yi, 1, ensembleSize) + tile(join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, ensembleSize) * randn(Y.dims(0), ensembleSize);
								else if (complexType == 2)
									Yi = tile(Yi, 1, ensembleSize) + tile(join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, ensembleSize) * randn(Y.dims(0), ensembleSize);
							}
						}
						else {
							if (complexType == 0)
								Y = m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)));
							else if (complexType == 3)
								Y = join(0, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))),
									imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))));
							else {
								Y = real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))));
								Yi = imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))));
							}
							//mexPrintf("Y.dims(0) = %d\n", Y.dims(0));
							//mexPrintf("Y.dims(1) = %d\n", Y.dims(1));
							//mexPrintf("Y.summa = %f\n", af::sum<float>(flat(Y)));
							//mexEvalString("pause(.0001);");
							if (sparseR) {
								Y = tile(Y, 1, ensembleSize) + matmul(*R1, randn(NmU, ensembleSize));
								if (complexType == 1)
									Yi = tile(Yi, 1, ensembleSize) + matmul(*R1, randn(NmU, ensembleSize));
								else if (complexType == 2)
									Yi = tile(Yi, 1, ensembleSize) + matmul(*R1i, randn(NmU, ensembleSize));
							}
							else {
								Y = tile(Y, 1, ensembleSize) + tile(*R1, 1, ensembleSize) * randn(Y.dims(0), ensembleSize);
								if (complexType == 1)
									Yi = tile(Yi, 1, ensembleSize) + tile(*R1, 1, ensembleSize) * randn(Y.dims(0), ensembleSize);
								else if (complexType == 2)
									Yi = tile(Yi, 1, ensembleSize) + tile(*R1i, 1, ensembleSize) * randn(Y.dims(0), ensembleSize);
							}
						}
						//mexPrintf("Y.dims(0) = %d\n", Y.dims(0));
						//mexPrintf("Y.dims(1) = %d\n", Y.dims(1));
						//mexPrintf("Y.summa = %f\n", af::sum<float>(flat(Y)));
						//mexEvalString("pause(.0001);");
					}
					const array A = X - tile(mean(X, 1), 1, X.dims(1));
					if (algorithm == 3) {
						//const array PP = matmulNT(A, A) / ensembleF;
						//mexPrintf("A.dims(0) = %d\n", A.dims(0));
						//mexPrintf("A.dims(1) = %d\n", A.dims(1));
						//mexEvalString("pause(.0001);");
						computeKG(algorithm, KG, HH, sparseR, R, A / ensembleS, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si);
						if (complexType == 3)
							if (useEnsembleMean)
								X = mean(X + matmul(KG, Y - join(0, matmul(S[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], X(seq(X.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], X(seq(X.dims(0) / 2, end), span)))), 1);
							else
								X = (X + matmul(KG, Y - join(0, matmul(S[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], X(seq(X.dims(0) / 2, end), span)),
									matmul(Si[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], X(seq(X.dims(0) / 2, end), span)))));
						else
							if (useEnsembleMean)
								X = mean(X + matmul(KG, Y - matmul(S[tt % hnU], X)), 1);
							else
								X = (X + matmul(KG, Y - matmul(S[tt % hnU], X)));
						//mexPrintf("X.dims(0) = %d\n", X.dims(0));
						//mexPrintf("X.dims(1) = %d\n", X.dims(1));
						//mexEvalString("pause(.0001);");
					}
					else if (algorithm == 4) {
						//mexPrintf("hnU = %d\n", hnU);
						//mexPrintf("sizeR = %d\n", sizeR);
						//mexPrintf("A.dims(0) = %d\n", A.dims(0));
						//mexPrintf("X.summa = %f\n", af::sum<float>(flat(X)));
						//mexEvalString("pause(.0001);");
						if (complexType == 3)
							HH = join(0, matmul(S[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], A(seq(A.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], A(seq(A.dims(0) / 2, end), span)));
						else
							HH = matmul(S[tt % hnU], A);
						//sync();
						//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
						//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
						//mexPrintf("HH.summa = %f\n", af::sum<float>(flat(HH)));
						//mexEvalString("pause(.0001);");
						PP = matmulNT(HH, HH) / ensembleF;
						//sync();
						//mexPrintf("PP.dims(0) = %d\n", PP.dims(0));
						//mexPrintf("PP.dims(1) = %d\n", PP.dims(1));
						//mexEvalString("pause(.0001);");
						if (sparseR)
							PP = PP + R[tt % sizeR];
						else {
							if (regularization == 1) {
								if (complexType == 3)
									PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR);
								else
									PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR);
							}
							else
								PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + R[tt % sizeR];
						}
						//mexPrintf("PP.summa = %f\n", af::sum<float>(flat(inverse(PP))));
						//mexEvalString("pause(.0001);");
						//sync();
						//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
						//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
						//mexEvalString("pause(.0001);");
						//SS = matmul(constant(0.f, Nm, Nm), Y - matmul(S[tt % hnU], X));
						//mexPrintf("SS.dims(0) = %d\n", SS.dims(0));
						//mexPrintf("SS.dims(1) = %d\n", SS.dims(1));
						//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
						//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
						//mexEvalString("pause(.0001);");
						//array D = (Y - matmul(S[tt % hnU], X));
						//array apu = solve(PP, D);
						//array apu2 = matmul(A, HH.T());
						//mexPrintf("apu.summa = %f\n", af::sum<float>(flat(apu)));
						//mexPrintf("apu2.summa = %f\n", af::sum<float>(flat(apu2)));
						//mexPrintf("D.summa = %f\n", af::sum<float>(flat(D)));
						//mexEvalString("pause(.0001);");
						if (complexType == 3)
							if (useEnsembleMean)
							X = mean(X + (1.f / ensembleF) * matmul(A, HH.T(), solve(PP, Y - join(0, matmul(S[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], X(seq(X.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], X(seq(X.dims(0) / 2, end), span))))), 1);
							else
								X = (X + (1.f / ensembleF) * matmul(A, HH.T(), solve(PP, Y - join(0, matmul(S[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], X(seq(X.dims(0) / 2, end), span)),
									matmul(Si[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], X(seq(X.dims(0) / 2, end), span))))));
						else
							if (useEnsembleMean)
								X = mean(X + (1.f / ensembleF) * matmul(A, HH.T(), solve(PP, Y - matmul(S[tt % hnU], X))), 1);
							else
								X = (X + (1.f / ensembleF) * matmul(A, HH.T(), solve(PP, Y - matmul(S[tt % hnU], X))));
						//mexPrintf("X2.summa = %f\n", af::sum<float>(flat(X)));
						//mexEvalString("pause(.0001);");
					}
					else if (algorithm == 5) {
						//mexPrintf("X.summa = %f\n", af::sum<float>(flat(X)));
						//mexEvalString("pause(.0001);");
						if (complexType == 3)
							HH = join(0, matmul(S[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], A(seq(A.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], A(seq(A.dims(0) / 2, end), span)));
						else
							HH = matmul(S[tt % hnU], A);
						//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
						//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
						//mexPrintf("HH.summa = %f\n", af::sum<float>(flat(HH)));
						//mexEvalString("pause(.0001);");
						if (sparseR)
							QQ = matmulTN(HH, matmul(R[tt % sizeR], HH / ensembleF));
						else {
							if (regularization == 1)
								if (complexType == 3)
									QQ = matmul(HH.T(), tile(join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH / ensembleF);
								else
									QQ = matmul(HH.T(), tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH / ensembleF);
							else
								QQ = matmul(HH.T(), tile(R[tt % sizeR], 1, HH.dims(1)) * HH / ensembleF);
						}
						if (DEBUG) {
							mexPrintf("QQ.dims(0) = %d\n", QQ.dims(0));
							mexPrintf("QQ.dims(1) = %d\n", QQ.dims(1));
							mexEvalString("pause(.0001);");
						}
						//QQ = identity(QQ.dims(0), QQ.dims(0)) + QQ;
						//mexPrintf("QQ.summa = %f\n", af::sum<float>(flat(QQ)));
						//mexPrintf("R[tt % sizeR].summa = %f\n", af::sum<float>(flat(R[tt % sizeR])));
						//mexEvalString("pause(.0001);");
						QQ(seq(0, end, QQ.dims(0) + 1)) = QQ(seq(0, end, QQ.dims(0) + 1)) + 1.f;
						array D;
						if (complexType == 3)
							D = Y - join(0, matmul(S[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], X(seq(X.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], X(seq(X.dims(0) / 2, end), span)));
						else
							D = (Y - matmul(S[tt % hnU], X));
						if (sparseR)
							Z = matmulTN(HH, matmul(R[tt % sizeR], D));
						else {
							if (regularization == 1)
								if (complexType == 3)
									Z = matmul(HH.T(), tile(join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR), 1, Y.dims(1)) * D);
								else
									Z = matmul(HH.T(), tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, Y.dims(1)) * D);
							else
								Z = matmulTN(HH, tile(R[tt % sizeR], 1, D.dims(1)) * D);
							//Z = matmulTN(HH, diag(R[tt % sizeR], 0, false));
						}
						choleskyInPlace(QQ);
						//mexPrintf("Z.dims(0) = %d\n", Z.dims(0));
						//mexPrintf("Z.dims(1) = %d\n", Z.dims(1));
						//mexPrintf("Z.summa = %f\n", af::sum<float>(flat(Z)));
						//mexEvalString("pause(.0001);");
						//const array M = identity(Z.dims(1), Z.dims(1)) - matmul(HH, solve(QQ, Z)) / ensembleF;
						//const array M = D - matmul(HH, solve(QQ, Z)) / ensembleF;
						const array M = D - matmul(HH, solve(QQ, solve(QQ.T(), Z, AF_MAT_LOWER), AF_MAT_UPPER)) / ensembleF;
						if (DEBUG) {
							mexPrintf("M.dims(0) = %d\n", M.dims(0));
							mexPrintf("M.dims(1) = %d\n", M.dims(1));
							mexPrintf("M.summa = %f\n", af::sum<float>(flat(M)));
							mexEvalString("pause(.0001);");
						}
						//M(seq(0, end, M.dims(0) + 1)) = M(seq(0, end, M.dims(0) + 1));
						if (sparseR)
							Z = matmulTN(HH, matmul(R[tt % sizeR], M));
						else
							if (regularization == 1)
								if (complexType == 3)
									Z = matmulTN(HH, tile(join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR), 1, M.dims(1)) * M);
								else
									Z = matmulTN(HH, tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, M.dims(1)) * M);
							else
								//Z = matmul(diag(R[tt % sizeR], 0, false), M);
								Z = matmulTN(HH, tile(R[tt % sizeR], 1, M.dims(1)) * M);
						//X = (X + matmul(A, Z) / ensembleF);
						//mexPrintf("Z.summa = %f\n", af::sum<float>(flat(Z)));
						//mexEvalString("pause(.0001);");
						//array apu = matmul(Z, D);
						//array apu2 = matmul(A, HH.T());
						//mexPrintf("apu.summa = %f\n", af::sum<float>(flat(apu)));
						//mexPrintf("apu2.summa = %f\n", af::sum<float>(flat(apu2)));
						//mexPrintf("D.summa = %f\n", af::sum<float>(flat(D)));
						//mexEvalString("pause(.0001);");
						//X = (X + matmul(A, HH.T(), Z, D) / ensembleF);
						if (useEnsembleMean)
							X = mean(X + matmul(A, Z) / ensembleF, 1);
						else
							X = (X + matmul(A, Z) / ensembleF);
						if (DEBUG) {
							mexPrintf("X2.summa = %f\n", af::sum<float>(flat(X)));
							mexEvalString("pause(.0001);");
						}
					}
					else if (algorithm == 6) {
						if (complexType == 3)
							HH = join(0, matmul(S[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], A(seq(A.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], A(seq(A.dims(0) / 2, end), span)));
						else
							HH = matmul(S[tt % hnU], A);
						array D;
						if (complexType == 3)
							D = Y - join(0, matmul(S[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], X(seq(X.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], X(seq(0, X.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], X(seq(X.dims(0) / 2, end), span)));
						else
							D = (Y - matmul(S[tt % hnU], X));
						if (DEBUG) {
							mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
							mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
							mexEvalString("pause(.0001);");
						}
						array B;
						if (sparseR) {
							SR = R[tt % sizeR];
							B = matmul(SR, HH);
							PP = matmul(SR, D - matmul(B, solve(identity(B.dims(1), B.dims(1)) + matmulTN(B, B) / ensembleF, matmulTN(B, matmul(SR, D, AF_MAT_TRANS)))) / ensembleF);
						}
						else {
							if (regularization == 1)
								if (complexType == 3)
									SR = join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR);
								else
									SR = join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR);
							else
								SR = R[tt % sizeR];
							B = tile(SR, 1, HH.dims(1)) * HH;
							//mexPrintf("B.dims(0) = %d\n", B.dims(0));
							//mexPrintf("B.dims(1) = %d\n", B.dims(1));
							//mexPrintf("SR.dims(0) = %d\n", SR.dims(0));
							//mexPrintf("SR.dims(1) = %d\n", SR.dims(1));
							//mexEvalString("pause(.0001);");
							array testi2 = matmulTN(B, B);
							//mexPrintf("testi2.dims(0) = %d\n", testi2.dims(0));
							//mexPrintf("testi2.dims(1) = %d\n", testi2.dims(1));
							//mexEvalString("pause(.0001);");
							array testi = matmul(B, solve(identity(B.dims(1), B.dims(1)) + matmulTN(B, B) / ensembleF, matmulTN(B, tile(SR, 1, D.dims(1)) * D))) / ensembleF;
							//mexPrintf("testi.dims(0) = %d\n", testi.dims(0));
							//mexPrintf("testi.dims(1) = %d\n", testi.dims(1));
							//mexEvalString("pause(.0001);");
							PP = D - matmul(B, solve(identity(B.dims(1), B.dims(1)) + matmulTN(B, B) / ensembleF, matmulTN(B, tile(SR, 1, D.dims(1)) * D))) / ensembleF;
							//mexPrintf("PP.dims(0) = %d\n", PP.dims(0));
							//mexPrintf("PP.dims(1) = %d\n", PP.dims(1));
							//mexEvalString("pause(.0001);");
							PP = PP * tile(SR, 1, PP.dims(1));
						}
						//mexPrintf("PP.dims(0) = %d\n", PP.dims(0));
						//mexPrintf("PP.dims(1) = %d\n", PP.dims(1));
						//mexEvalString("pause(.0001);");
						if (useEnsembleMean)
							X = mean(X + matmul(A, HH.T(), PP) / ensembleF, 1);
						else
							X = (X + matmul(A, HH.T(), PP) / ensembleF);
					}
					if (complexType == 1 || complexType == 2) {
						const array Ai = Xi - tile(mean(Xi, 1), 1, Xi.dims(1));
						if (algorithm == 3) {
							//const array PPi = matmulNT(Ai, Ai) / ensembleF;
							if (complexType == 2)
								if (complexS)
									computeKG(algorithm, KG, HH, sparseR, Ri, Ai / ensembleS, Si, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si);
								else
									computeKG(algorithm, KG, HH, sparseR, Ri, Ai / ensembleS, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si);
							else
								computeKG(algorithm, KG, HH, sparseR, R, Ai / ensembleS, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si);
							if (useEnsembleMean)
								Xi = mean(Xi + matmul(KG, Yi - matmul(S[tt % hnU], Xi)), 1);
							else
								Xi = (Xi + matmul(KG, Yi - matmul(S[tt % hnU], Xi)));
						}
						else if (algorithm == 4) {
							//mexPrintf("Ai.dims(0) = %d\n", Ai.dims(0));
							//mexPrintf("Ai.dims(1) = %d\n", Ai.dims(1));
							//mexEvalString("pause(.0001);");
							if (complexS)
								HH = matmul(Si[tt % hnU], Ai);
							else
								HH = matmul(S[tt % hnU], Ai);
							PP = matmulNT(HH, HH) / ensembleF;
							//mexPrintf("PPi.dims(0) = %d\n", PP.dims(0));
							//mexPrintf("PPi.dims(1) = %d\n", PP.dims(1));
							//mexEvalString("pause(.0001);");
							if (sparseR)
								if (complexType == 1)
									PP = PP + R[tt % sizeR];
								else
									PP = PP + Ri[tt % sizeR];
							else {
								if (regularization == 1) {
									if (complexType == 1)
										PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR);
									else
										PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR);
								}
								else
									if (complexType == 1)
										PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + R[tt % sizeR];
									else
										PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + Ri[tt % sizeR];
							}
							//mexPrintf("PPi2.dims(0) = %d\n", PP.dims(0));
							//mexPrintf("PPi2.dims(1) = %d\n", PP.dims(1));
							//mexEvalString("pause(.0001);");
							if (complexS)
								if (useEnsembleMean)
									Xi = (Xi + (1.f / ensembleF) * matmul(Ai, HH.T(), solve(PP, Yi - matmul(Si[tt % hnU], Xi))));
								else
									Xi = mean(Xi + (1.f / ensembleF) * matmul(Ai, HH.T(), solve(PP, Yi - matmul(Si[tt % hnU], Xi))), 1);
							else
								if (useEnsembleMean)
									Xi = mean(Xi + (1.f / ensembleF) * matmul(Ai, HH.T(), solve(PP, Yi - matmul(S[tt % hnU], Xi))), 1);
								else
									Xi = (Xi + (1.f / ensembleF) * matmul(Ai, HH.T(), solve(PP, Yi - matmul(S[tt % hnU], Xi))));
						}
						else if (algorithm == 5) {
							if (complexS)
								HH = matmul(Si[tt % hnU], Ai);
							else
								HH = matmul(S[tt % hnU], Ai);
							//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
							//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
							//mexEvalString("pause(.0001);");
							if (sparseR)
								QQ = matmulTN(HH, matmul(R[tt % sizeR], HH));
							else {
								if (regularization == 1)
									if (complexType == 1)
										QQ = matmul(HH.T(), tile(join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
									else
										QQ = matmul(HH.T(), tile(join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
								else
									if (complexType == 1)
										QQ = matmul(HH.T(), tile(R[tt % sizeR], 1, HH.dims(1)) * HH);
									else
										QQ = matmul(HH.T(), tile(Ri[tt % sizeR], 1, HH.dims(1)) * HH);
							}
							//QQ(seq(0, end, QQ.dims(0) + 1)) = QQ(seq(0, end, QQ.dims(0) + 1)) + ensembleF;
							QQ = identity(QQ.dims(0), QQ.dims(0)) + QQ * (1.f / ensembleF);
							//mexPrintf("QQ.dims(0) = %d\n", QQ.dims(0));
							//mexPrintf("QQ.dims(1) = %d\n", QQ.dims(1));
							//mexEvalString("pause(.0001);");
							//array QA;
							//cholesky(QA, QQ);
							array D;
							if (complexS)
								D = (Yi - matmul(Si[tt % hnU], Xi));
							else
								D = (Yi - matmul(S[tt % hnU], Xi));
							if (regularization == 1)
								if (complexType == 1)
									Z = matmul(HH.T(), tile(join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, D.dims(1)) * D);
								else
									Z = matmul(HH.T(), tile(join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, D.dims(1)) * D);
							else
								if (complexType == 1)
									Z = matmul(HH.T(), tile(R[tt % sizeR], 1, D.dims(1)) * D);
								else
									Z = matmul(HH.T(), tile(Ri[tt % sizeR], 1, D.dims(1)) * D);
							//mexPrintf("Z.dims(0) = %d\n", Z.dims(0));
							//mexPrintf("Z.dims(1) = %d\n", Z.dims(1));
							//mexEvalString("pause(.0001);");
							const array M = D - matmul(HH, solve(QQ, Z)) * (1.f / ensembleF);
							//array M = matmul(HH, solve(QA, solve(QA.T(), Z, AF_MAT_LOWER), AF_MAT_UPPER));
							//M = identity(M.dims(0)) - M;
							//M(seq(0, end, M.dims(0) + 1)) = M(seq(0, end, M.dims(0) + 1)) + 1.f;
							//mexPrintf("M.dims(0) = %d\n", M.dims(0));
							//mexPrintf("M.dims(1) = %d\n", M.dims(1));
							//mexEvalString("pause(.0001);");
							if (regularization == 1)
								if (complexType == 1)
									Z = matmulTN(HH, tile(join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, M.dims(1)) * M);
								else
									Z = matmulTN(HH, tile(join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, M.dims(1)) * M);
							else
								if (complexType == 1)
									Z = matmulTN(HH, tile(R[tt % sizeR], 1, M.dims(1)) * M);
								else
									Z = matmulTN(HH, tile(Ri[tt % sizeR], 1, M.dims(1)) * M);
							//mexPrintf("Z.dims(0) = %d\n", Z.dims(0));
							//mexPrintf("Z.dims(1) = %d\n", Z.dims(1));
							//mexEvalString("pause(.0001);");
							if (useEnsembleMean)
								Xi = mean(Xi + (1.f / ensembleF) * matmul(Ai, Z), 1);
							else
								Xi = (Xi + (1.f / ensembleF) * matmul(Ai, Z));
						}
						else if (algorithm == 6) {
							if (complexS)
								HH = matmul(Si[tt % hnU], Ai);
							else
								HH = matmul(S[tt % hnU], Ai);
							array D;
							if (complexS)
								D = (Yi - matmul(Si[tt % hnU], Xi));
							else
								D = (Yi - matmul(S[tt % hnU], Xi));
							array B;
							if (sparseR) {
								if (complexType == 1)
									SR = R[tt % sizeR];
								else
									SR = Ri[tt % sizeR];
								B = matmul(SR, HH);
								PP = matmul(SR, D - matmul(B, solve(identity(B.dims(1), B.dims(1)) + matmulTN(B, B) / ensembleF, matmulTN(B, matmul(SR, D, AF_MAT_TRANS)))) / ensembleF, AF_MAT_TRANS);
							}
							else {
								if (regularization == 1)
									if (complexType == 1)
										SR = join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR);
									else
										SR = join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR);
								else
									if (complexType == 1)
										SR = R[tt % sizeR];
									else
										SR = Ri[tt % sizeR];
								B = tile(SR, 1, HH.dims(1)) * HH;
								PP = D - matmul(B, solve(identity(B.dims(1), B.dims(1)) + matmulTN(B, B) / ensembleF, matmulTN(B, tile(SR, 1, D.dims(1)) * D))) / ensembleF;
								PP = tile(SR.T(), PP.dims(1), 1) * PP;
							}
							//const array PP = tile(SR, 1, Yi.dims(1)) * (Yi - matmul(S[tt % hnU], Xi)) - (1.f / ensembleF) *
							//	matmul(B, solve(diag(constant(1.f, B.dims(1))) + (1.f / ensembleF) * matmulTN(B, B), B.T()), Yi - matmul(S[tt % hnU], Xi)) * tile(SR.T(), Yi.dims(0), 1);
							if (useEnsembleMean)
								Xi = mean(Xi + matmul(Ai, HH.T(), PP, D) / ensembleF, 1);
							else
								Xi = (Xi + matmul(Ai, HH.T(), PP, D) / ensembleF);
						}
						if (useEnsembleMean)
							xt(span, tt + 1, NN) = complex(X, Xi);
						else
							xt(span, tt + 1, NN) = complex(mean(X, 1), mean(Xi, 1));
					}
					else if (complexType == 3)
						if (kineticModel)
							if (useEnsembleMean)
								xt(span, tt + 1, NN) = complex(join(0, X(seq(0, imDim - 1)), X(seq(imDimU, imDimU + imDim - 1))), join(0, X(seq(imDim, imDimU - 1)), X(seq(imDimU + imDim, end))));
							else {
								const array XT = mean(X, 1);
								xt(span, tt + 1, NN) = complex(join(0, XT(seq(0, imDim - 1)), XT(seq(imDimU, imDimU + imDim - 1))), join(0, XT(seq(imDim, imDimU - 1)), XT(seq(imDimU + imDim, end))));
							}
						else
							if (useEnsembleMean)
								xt(span, tt + 1, NN) = complex(X(seq(0, X.dims(0) / 2 - 1)), X(seq(X.dims(0) / 2, end)));
							else {
								const array XT = mean(X, 1);
								xt(span, tt + 1, NN) = complex(XT(seq(0, X.dims(0) / 2 - 1)), XT(seq(X.dims(0) / 2, end)));
							}
					else {
						if (useEnsembleMean)
							xt(span, tt + 1, NN) = X;
						else
							xt(span, tt + 1, NN) = mean(X, 1);
					}
					if (regularization > 2) {
						computeDenoising(xt, imDim, tt, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
							huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, Type);
						if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
							mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
							break;
						}
					}
				}
				else if (algorithm >= 7) {
					if (algorithm == 7) {
						const array A = X - tile(mean(X, 1), 1, X.dims(1));
						if (complexType == 3)
							HH = join(0, matmul(S[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], A(seq(A.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], A(seq(0, A.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], A(seq(A.dims(0) / 2, end), span)));
						else
							HH = matmul(S[tt % hnU], A);
						//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
						//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
						//mexEvalString("pause(.0001);");
						array w;
						if (sparseR)
							PP = matmulTN(HH, matmul(R[tt % sizeR], HH));
						else
							if (regularization == 1) {
								if (complexType == 3)
									PP = matmulTN(HH, tile(join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
								else
									PP = matmulTN(HH, tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
							}
							else
								PP = matmulTN(HH, tile(R[tt % sizeR], 1, HH.dims(1)) * HH);
						PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + fadingAlpha * ensembleF;
						//mexPrintf("PP.dims(0) = %d\n", PP.dims(0));
						//mexPrintf("PP.dims(1) = %d\n", PP.dims(1));
						//mexEvalString("pause(.0001);");
						af::svd(U, SS, V, PP);
						//mexPrintf("SS.dims(0) = %d\n", SS.dims(0));
						//mexPrintf("SS.dims(1) = %d\n", SS.dims(1));
						//mexEvalString("pause(.0001);");
						const array M = std::sqrt(ensembleF) * matmul(U * tile(1.f / sqrt(SS.T()), U.dims(0), 1), U.T());
						//mexPrintf("M.dims(0) = %d\n", M.dims(0));
						//mexPrintf("M.dims(1) = %d\n", M.dims(1));
						//mexEvalString("pause(.0001);");
						if (complexType == 3)
							if (regularization == 1)
								w = computeWL(U, SS, HH, R, m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))), Ly, X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							else
								w = computeW(U, SS, HH, R, m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))), X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
						else
							if (regularization == 1)
								w = computeWL(U, SS, HH, R, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							else
								w = computeW(U, SS, HH, R, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
						//X = mean(tile(mean(X, 1), 1, X.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M), 1);
						//mexPrintf("w.dims(0) = %d\n", w.dims(0));
						//mexPrintf("w.dims(1) = %d\n", w.dims(1));
						//mexEvalString("pause(.0001);");
						if (useSmoother) {
							GS[tt % N_lag] = fadingAlpha * matmul(identity(ensembleSize, ensembleSize) - constant(ensembleD2, ensembleSize, ensembleSize), tile(w, 1, M.dims(1)) + M) + ensembleD2;
							if (kineticModel)
								XS[tt % N_lag] = XSk(seq(0, imDimU - 1), span);
							else
								XS[tt % N_lag] = XSk;
							XS[tt % N_lag] = XSk;
							XSk = tile(mean(X, 1), 1, X.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M);
							if (useEnsembleMean)
								X = mean(XSk, 1);
							else
								X = XSk;
							//GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) = GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) + 1.f;
						}
						else
							if (useEnsembleMean)
								X = mean(tile(mean(X, 1), 1, X.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M), 1);
							else
								X = (tile(mean(X, 1), 1, X.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M));
					}
					else if (algorithm == 8) {
						const array LX = matmul(X, AA);
						if (complexType == 3)
							HH = join(0, matmul(S[tt % hnU], LX(seq(0, LX.dims(0) / 2 - 1), span)) - matmul(Si[tt % hnU], LX(seq(LX.dims(0) / 2, end), span)),
								matmul(Si[tt % hnU], LX(seq(0, LX.dims(0) / 2 - 1), span)) + matmul(S[tt % hnU], LX(seq(LX.dims(0) / 2, end), span)));
						else
							HH = matmul(S[tt % hnU], LX);
						array w;
						//mexPrintf("HH.dims(0) = %d\n", HH.dims(0));
						//mexPrintf("HH.dims(1) = %d\n", HH.dims(1));
						//mexEvalString("pause(.0001);");
						if (sparseR)
							PP = matmulTN(HH, matmul(R[tt % sizeR], HH));
						else
							if (regularization == 1)
								if (complexType == 3)
									PP = matmulTN(HH, tile(join(0, R[tt % sizeR](seq(0, R[tt % sizeR].dims(0) / 2 - 1)), constant(1.f, X.dims(0), 1) * RR,
										R[tt % sizeR](seq(R[tt % sizeR].dims(0) / 2, end)), constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
								else
									PP = matmulTN(HH, tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
							else
								PP = matmulTN(HH, tile(R[tt % sizeR], 1, HH.dims(1)) * HH);
						PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + fadingAlpha * ensembleF;
						//mexPrintf("PP.dims(0) = %d\n", PP.dims(0));
						//mexPrintf("PP.dims(1) = %d\n", PP.dims(1));
						//mexEvalString("pause(.0001);");
						af::svd(U, SS, V, PP);
						//mexPrintf("SS.dims(0) = %d\n", SS.dims(0));
						//mexPrintf("SS.dims(1) = %d\n", SS.dims(1));
						//mexEvalString("pause(.0001);");
						//const array M = std::sqrt(ensembleF) * matmul(batchFunc(U, 1.f / sqrt(SS.T()), batchMul), U.T(), AA.T());
						const array M = std::sqrt(ensembleF) * matmul(U * tile(1.f / sqrt(SS.T()), U.dims(0), 1), U.T(), AA.T());
						//mexPrintf("M.dims(0) = %d\n", M.dims(0));
						//mexPrintf("M.dims(1) = %d\n", M.dims(1));
						//mexEvalString("pause(.0001);");
						if (complexType == 3)
							if (regularization == 1)
								w = computeWL(U, SS, HH, R, m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))), Ly, X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							else
								w = computeW(U, SS, HH, R, m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL))), X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
						else
							if (regularization == 1)
								w = computeWL(U, SS, HH, R, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							else
								w = computeW(U, SS, HH, R, real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), X, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
						//mexPrintf("w.dims(0) = %d\n", w.dims(0));
						//mexPrintf("w.dims(1) = %d\n", w.dims(1));
						//mexEvalString("pause(.0001);");
						if (useSmoother) {
							GS[tt % N_lag] = fadingAlpha * matmul(AA, tile(w, 1, M.dims(1)) + M) + ensembleD2;
							if (kineticModel)
								XS[tt % N_lag] = XSk(seq(0, imDimU - 1), span);
							else
								XS[tt % N_lag] = XSk;
							XSk = tile(mean(X, 1), 1, X.dims(1)) + matmul(LX, tile(w, 1, M.dims(1)) + M);
							if (useEnsembleMean)
								X = mean(XSk, 1);
							else
								X = XSk;
							//GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) = GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) + 1.f;
						}
						else
							if (useEnsembleMean)
								X = mean(tile(mean(X, 1), 1, X.dims(1)) + matmul(LX, tile(w, 1, M.dims(1)) + M), 1);
							else
								X = (tile(mean(X, 1), 1, X.dims(1)) + matmul(LX, tile(w, 1, M.dims(1)) + M));
						if (DEBUG) {
							mexPrintf("X.dims(0) = %d\n", X.dims(0));
							mexPrintf("X.dims(1) = %d\n", X.dims(1));
							mexEvalString("pause(.0001);");
						}
					}
					if (complexType == 1 || complexType == 2) {
						if (algorithm == 7) {
							const array A = Xi - tile(mean(Xi, 1), 1, Xi.dims(1));
							if (complexS)
								HH = matmul(Si[tt % hnU], A);
							else
								HH = matmul(S[tt % hnU], A);
							array w;
							if (sparseR)
								if (complexType == 2)
									PP = matmulTN(HH, matmul(Ri[tt % sizeR], HH));
								else
									PP = matmulTN(HH, matmul(R[tt % sizeR], HH));
							else
								if (complexType == 1)
									if (regularization == 1)
										PP = matmulTN(HH, tile(join(0, R[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
									else
										PP = matmulTN(HH, tile(R[tt % sizeR], 1, HH.dims(1)) * HH);
								else
									if (regularization == 1)
										PP = matmulTN(HH, tile(join(0, Ri[tt % sizeR], constant(1.f, Xi.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
									else
										PP = matmulTN(HH, tile(Ri[tt % sizeR], 1, HH.dims(1)) * HH);
							PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + fadingAlpha * ensembleF;
							af::svd(U, SS, V, PP);
							const array M = std::sqrt(ensembleF) * matmul(U * tile(1.f / sqrt(SS.T()), U.dims(0), 1), U.T());
							if (complexRef && augType > 0 && regularization == 1) {
								if (complexType == 2)
									if (complexS)
										w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li, Xi, Si, tt, sizeR, hnU, sparseR, RR, complexType, Si);
									else
										w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
								else
									w = computeWL(U, SS, HH, R, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							}
							else {
								if (complexType == 2)
									if (complexS)
										if (regularization == 1)
											w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, Xi, Si, tt, sizeR, hnU, sparseR, RR, complexType, Si);
										else
											w = computeW(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Xi, Si, tt, sizeR, hnU, sparseR, RR, complexType, Si);
									else
										if (regularization == 1)
											w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
										else
											w = computeW(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
								else
									if (regularization == 1)
										w = computeWL(U, SS, HH, R, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
									else
										w = computeW(U, SS, HH, R, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							}
							if (useSmoother) {
								GSi[tt % N_lag] = fadingAlpha * matmul(identity(ensembleSize, ensembleSize) - constant(ensembleD2, ensembleSize, ensembleSize), tile(w, 1, M.dims(1)) + M) + ensembleD2;
								if (kineticModel)
									XSi[tt % N_lag] = XSki(seq(0, imDimU - 1), span);
								else
									XSi[tt % N_lag] = XSki;
								XSi[tt % N_lag] = XSki;
								XSki = tile(mean(Xi, 1), 1, Xi.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M);
								if (useEnsembleMean)
									Xi = mean(XSki, 1);
								else
									Xi = XSki;
								//GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) = GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) + 1.f;
							}
							else
								if (useEnsembleMean)
									Xi = tile(mean(Xi, 1), 1, Xi.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M);
								else
									Xi = mean(tile(mean(Xi, 1), 1, Xi.dims(1)) + matmul(A, tile(w, 1, M.dims(1)) + M), 1);
						}
						else if (algorithm == 8) {
							const array LX = matmul(Xi, AA);
							if (complexS)
								HH = matmul(Si[tt % hnU], LX);
							else
								HH = matmul(S[tt % hnU], LX);
							array w;
							if (sparseR)
								if (complexType == 2)
									PP = matmulTN(HH, matmul(Ri[tt % sizeR], HH));
								else
									PP = matmulTN(HH, matmul(R[tt % sizeR], HH));
							else
								if (complexType == 1)
									if (regularization == 1)
										PP = matmulTN(HH, tile(join(0, R[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
									else
										PP = matmulTN(HH, tile(R[tt % sizeR], 1, HH.dims(1)) * HH);
								else
									if (regularization == 1)
										PP = matmulTN(HH, tile(join(0, Ri[tt % sizeR], constant(1.f, X.dims(0), 1) * RR), 1, HH.dims(1)) * HH);
									else
										PP = matmulTN(HH, tile(Ri[tt % sizeR], 1, HH.dims(1)) * HH);
							PP(seq(0, end, PP.dims(0) + 1)) = PP(seq(0, end, PP.dims(0) + 1)) + fadingAlpha * ensembleF;
							af::svd(U, SS, V, PP);
							const array M = std::sqrt(ensembleF) * matmul(U * tile(1.f / sqrt(SS.T()), U.dims(0), 1), U.T(), AA.T());
							//mexPrintf("M.dims(0) = %d\n", M.dims(0));
							//mexPrintf("M.dims(1) = %d\n", M.dims(1));
							//mexEvalString("pause(.0001);");
							if (complexRef && augType > 0 && regularization == 1) {
								if (complexType == 2)
									if (complexS)
										w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li, Xi, Si, tt, sizeR, hnU, sparseR, RR, complexType, Si);
									else
										w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
								else
									w = computeWL(U, SS, HH, R, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Li, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							}
							else {
								if (complexType == 2)
									if (complexS)
										if (regularization == 1)
											w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, Xi, Si, tt, sizeR, hnU, sparseR, RR, complexType, Si);
										else
											w = computeW(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Xi, Si, tt, sizeR, hnU, sparseR, RR, complexType, Si);
									else
										if (regularization == 1)
											w = computeWL(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
										else
											w = computeW(U, SS, HH, Ri, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
								else
									if (regularization == 1)
										w = computeWL(U, SS, HH, R, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Ly, Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
									else
										w = computeW(U, SS, HH, R, imag(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))), Xi, S, tt, sizeR, hnU, sparseR, RR, complexType, Si);
							}
							//mexPrintf("w.dims(0) = %d\n", w.dims(0));
							//mexPrintf("w.dims(1) = %d\n", w.dims(1));
							//mexEvalString("pause(.0001);");
							if (useSmoother) {
								GSi[tt % N_lag] = fadingAlpha * matmul(AA, tile(w, 1, M.dims(1)) + M) + ensembleD2;
								if (kineticModel)
									XSi[tt % N_lag] = XSki(seq(0, imDimU - 1), span);
								else
									XSi[tt % N_lag] = XSki;
								XSi[tt % N_lag] = XSki;
								XSki = tile(mean(Xi, 1), 1, Xi.dims(1)) + matmul(LX, tile(w, 1, M.dims(1)) + M);
								if (useEnsembleMean)
									Xi = mean(XSki, 1);
								else
									Xi = XSki;
								//GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) = GSi[tt % N_lag](seq(0, GSi[tt % N_lag].dims(0) + 1, end)) + 1.f;
							}
							else
								if (useEnsembleMean)
									Xi = mean(tile(mean(Xi, 1), 1, Xi.dims(1)) + matmul(LX, tile(w, 1, M.dims(1)) + M), 1);
								else
									Xi = tile(mean(Xi, 1), 1, Xi.dims(1)) + matmul(LX, tile(w, 1, M.dims(1)) + M);
							if (DEBUG) {
								mexPrintf("Xi.dims(0) = %d\n", Xi.dims(0));
								mexPrintf("Xi.dims(1) = %d\n", Xi.dims(1));
								mexEvalString("pause(.0001);");
							}
						}
						if (useEnsembleMean)
							xt(span, tt + 1, NN) = complex(X, Xi);
						else
							xt(span, tt + 1, NN) = complex(mean(X, 1), mean(Xi, 1));
					}
					else if (complexType == 3) {
						if (kineticModel)
							if (useEnsembleMean)
								xt(span, tt + 1, NN) = complex(join(0, X(seq(0, imDim - 1)), X(seq(imDimU, imDimU + imDim - 1))), join(0, X(seq(imDim, imDimU - 1)), X(seq(imDimU + imDim, end))));
							else {
								const array XT = mean(X, 1);
								xt(span, tt + 1, NN) = complex(join(0, XT(seq(0, imDim - 1)), XT(seq(imDimU, imDimU + imDim - 1))), join(0, XT(seq(imDim, imDimU - 1)), XT(seq(imDimU + imDim, end))));
							}
						else
							if (useEnsembleMean)
								xt(span, tt + 1, NN) = complex(X(seq(0, X.dims(0) / 2 - 1)), X(seq(X.dims(0) / 2, end)));
							else {
								const array XT = mean(X, 1);
								xt(span, tt + 1, NN) = complex(XT(seq(0, X.dims(0) / 2 - 1)), XT(seq(X.dims(0) / 2, end)));
							}
					}
					else {
						if (useEnsembleMean)
							xt(span, tt + 1, NN) = X;
						else
							xt(span, tt + 1, NN) = mean(X, 1);
					}
					if (regularization > 2) {
						computeDenoising(xt, imDim, tt, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
							huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, Type);
						if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
							mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
							break;
						}
					}
					if (useSmoother && ((tt == N_lag - 1) || (tt >= N_lag - 1 && jg == skip) || tt == (Nt - 1))) {
						jg = 0;
						array XSapu = constant(0.f, XS[0].dims(0), N_lag);
						//array GSapu = constant(1.f, GS[0].dims(0), GS[0].dims(1));
						array GSapu = identity(GS[0].dims(0), GS[0].dims(1));
						uint64_t ls = tt % N_lag;
						for (int64_t to = N_lag - 1; to >= 0LL; to--) {
							//GSapu *= GS[ls];
							GSapu = matmul(GSapu, GS[ls]);
							XSapu(span, to) = mean(matmul(XS[ls], GSapu), 1);
							ls++;
							if (ls == N_lag)
								ls = 0ULL;
						}
						if (complexType == 1 || complexType == 2) {
							array XSapui = constant(0.f, XSi[0].dims(0), N_lag);
							//array GSapui = constant(1.f, GSi[0].dims(0), GSi[0].dims(1));
							array GSapu = identity(GSi[0].dims(0), GSi[0].dims(1));
							ls = tt % N_lag;
							for (int64_t to = N_lag - 1; to >= 0; to--) {
								//GSapui *= GSi[ls];
								GSapu = matmul(GSapu, GSi[ls]);
								XSapui(span, to) = mean(matmul(XSi[ls], GSapu), 1);
								ls++;
								if (ls == N_lag)
									ls = 0;
							}
							xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = complex(XSapu, XSapui);
						}
						else if (complexType == 3)
							xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = complex(XSapu(seq(0, XSapu.dims(0) / 2 - 1)), XSapu(seq(XSapu.dims(0) / 2, end)));
						else
							xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = XSapu;
						if (DEBUG) {
							mexPrintf("tt + 1LL - (N_lag) = %d\n", tt + 1LL - (N_lag));
						}
					}
				}
				if (DEBUG) {
					mexPrintf("tt = %d\n", tt);
				}
			}
		}
		else if (algorithm == 9) {
			array Cred, Credi, A, Ai, B, Bi, alphaEst, PH, PHi;
			if (Type == 0)
				Type = 2;
			uint64_t kk = 0ULL;
			if (useSmoother) {
				KS.resize(N_lag);
				if (complexType == 2)
					KSi.resize(N_lag);
				if (complexType == 0 || complexType == 3)
					xlt1 = constant(0.f, imDimU, N_lag + 1);
				else
					xlt1 = constant(0.f, imDimU, N_lag + 1, c32);
			}
			if (complexType == 3) {
				PH = matmul3(S, Si, Pred, 0);
			}
			else
				if (complexS)
					PH = matmul(Si[0], Pred);
				else
					PH = matmul(S[0], Pred);
			//mexPrintf("PH.dims(0) = %d\n", PH.dims(0));
			//mexPrintf("PH.dims(1) = %d\n", PH.dims(1));
			//mexEvalString("pause(.0001);");
			//Pplus = inverse(matmulTN(PH, tile(R[0], 1, PH.dims(1)) * PH) + diag(Pplus, 0, false));
			//Pplus = inverse(matmulTN(PH, tile(R[0], 1, PH.dims(1)) * PH) + identity(PH.dims(1), PH.dims(1)));
			//Pplus = (matmulTN(PH, tile(R[0], 1, PH.dims(1)) * PH));
			//mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
			//mexPrintf("Pred.summa = %f\n", af::sum<float>(flat(Pred)));
			//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
			//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
			//mexPrintf("Pred.dims(0) = %d\n", Pred.dims(0));
			//mexPrintf("Pred.dims(1) = %d\n", Pred.dims(1));
			//mexEvalString("pause(.0001);");
			if (!useSmoother && regularization != 3) {
				//Pplus = diag(Pplus, 0, false);
				//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
				//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
				//mexEvalString("pause(.0001);");
				int success = cholesky(A, Pplus);
				if (success != 0) {
					mexPrintf("Cholesky failed\n");
					return;
				}
				else if (DEBUG)
					mexPrintf("Cholesky succeeded\n");
				//mexPrintf("A.summa = %f\n", af::sum<float>(flat(A)));
				//mexEvalString("pause(.0001);");
				//A = inverse(A);
				if (complexType == 2)
					cholesky(Ai, Pplusi);
				//mexPrintf("A.summa = %f\n", af::sum<float>(flat(A)));
				//mexEvalString("pause(.0001);");
			}
			else
				Pplus = 1.f / Pplus;
			if (covIter > 0 || steadyKF) {
				while (!steady) {
					loadSystemMatrix(storeData, sparseS, kk, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
						sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
					if (complexType == 3) {
						PH = matmul3(S, Si, Pred, kk % hnU);
					}
					else
						if (complexS)
							PH = matmul(Si[kk % hnU], Pred);
						else
							PH = matmul(S[kk % hnU], Pred);
					if (useF) {
						if (!useSmoother && regularization != 3)
							//if (useKineticModel) {
							//	B = transpose(solve(A.T(), transpose(Pred), AF_MAT_UPPER));
							//	B(seq(0, B.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], B);
							//}
							//else
							B = transpose(solve(A, transpose(matmul(F[kk % sizeF], Pred)), AF_MAT_LOWER));
						else
							//if (useKineticModel) {
							//	B = matmul(Pred, A);
							//	B(seq(0, B.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], B);
							//}
							//else
							B = matmul(matmul(F[kk % sizeF], Pred), A);
						if (complexType == 2) {
							if (!useSmoother && regularization != 3)
								//if (useKineticModel) {
								//	Bi = transpose(solve(Ai.T(), transpose(Pred), AF_MAT_UPPER));
								//	Bi(seq(0, Bi.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], Bi);
								//}
								//else
								if (complexF)
									Bi = transpose(solve(Ai, transpose(matmul(Fi[kk % sizeF], Pred)), AF_MAT_LOWER));
								else
									Bi = transpose(solve(Ai, transpose(matmul(F[kk % sizeF], Pred)), AF_MAT_LOWER));
							else
								//if (useKineticModel) {
								//	Bi = transpose(solve(Ai.T(), transpose(Pred), AF_MAT_UPPER));
								//	Bi(seq(0, Bi.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], Bi);
								//}
								//else
								if (complexF)
									Bi = matmul(matmul(Fi[kk % sizeF], Pred), Ai);
								else
									Bi = matmul(matmul(F[kk % sizeF], Pred), Ai);
						}
					}
					else {
						if (!useSmoother && regularization != 3)
							B = transpose(solve(A, Pred.T(), AF_MAT_LOWER));
						else
							B = matmul(Pred, A);
						if (complexType == 2) {
							if (!useSmoother && regularization != 3)
								Bi = transpose(solve(Ai, Pred.T(), AF_MAT_LOWER));
							else
								Bi = matmul(Pred, Ai);
						}
					}
					if (sparseQ) {
						Cred = Q[kk % sizeQ] - matmul(matmul(Q[kk % sizeQ], B), solve(matmulTN(B, matmul(Q[kk % sizeQ], B)) + identity(B.dims(1)), transpose(matmul(Q[kk % sizeQ], B))));
						if (complexType == 2)
							Credi = Qi[kk % sizeQ] - matmul(matmul(Qi[kk % sizeQ], Bi), solve(matmulTN(Bi, matmul(Qi[kk % sizeQ], Bi)) + identity(Bi.dims(1)), transpose(matmul(Qi[kk % sizeQ], Bi))));
					}
					else {
						Cred = tile(Q[kk % sizeQ](span, NN), 1, Pred.dims(1)) * Pred - matmul(tile(Q[kk % sizeQ](span, NN), 1, B.dims(1)) * B, solve(matmulTN(B, tile(Q[kk % sizeQ](span, NN), 1, B.dims(1)) * B) + identity(B.dims(1), B.dims(1)), matmulTN(B, tile(Q[kk % sizeQ](span, NN), 1, Pred.dims(1)) * Pred)));
						if (complexType == 2) {
							Credi = matmul(tile(Qi[kk % sizeQ](span, NN), 1, Bi.dims(1)) * Bi, solve(matmulTN(Bi, tile(Qi[kk % sizeQ](span, NN), 1, Bi.dims(1)) * Bi) + identity(B.dims(1), B.dims(1)), Bi.T() * tile(Qi[kk % sizeQ](span, NN).T(), Bi.dims(1), 1)));
							Credi(seq(0, Credi.dims(0) + 1, end)) = Credi(seq(0, Credi.dims(0) + 1, end)) + Qi[kk % sizeQ](span, NN);
						}
					}
					if (sparseR) {
						if (!useSmoother && regularization != 3) {
							Pplus = matmulTN(PH, matmul(R[kk % sizeR], PH)) + identity(Pplus.dims(0), Pplus.dims(0)) + matmulTN(Pred, Cred);
							cholesky(A, Pplus, false);
						}
						else {
							Pplus = inverse(matmulTN(PH, matmul(R[kk % sizeR], PH)) + identity(Pplus.dims(0), Pplus.dims(0)) + matmulTN(Pred, Cred));
						}
						if (complexType == 1 || complexType == 2) {
							if (!useSmoother && regularization != 3) {
								if (complexType == 2) {
									Pplusi = matmulTN(PH, matmul(Ri[kk % sizeR], PH)) + identity(Pplusi.dims(0), Pplusi.dims(0)) + matmulTN(Pred, Credi);
									cholesky(Ai, Pplusi, false);
								}
							}
							else {
								if (complexType == 2) {
									Pplusi = inverse(matmulTN(PH, matmul(Ri[kk % sizeR], PH)) + identity(Pplusi.dims(0), Pplusi.dims(0)) + matmulTN(Pred, Credi));
								}
							}
						}
					}
					else {
						if (!useSmoother && regularization != 3) {
							Pplus = matmul(PH.T(), tile(R[kk % sizeR], 1, PH.dims(1)) * PH) + matmulTN(Pred, Cred);
							cholesky(A, Pplus, false);
						}
						else {
							Pplus = inverse(matmul(PH.T(), tile(R[kk % sizeR], 1, PH.dims(1)) * PH) + matmulTN(Pred, Cred));
						}
						if (complexType == 2) {
							if (!useSmoother && regularization != 3) {
								Pplusi = matmul(PH.T(), tile(Ri[kk % sizeR], 1, PH.dims(1)) * PH) + matmulTN(Pred, Credi);
								cholesky(Ai, Pplus, false);
							}
							else {
								Pplusi = inverse(matmul(PH.T(), tile(Ri[kk % sizeR], 1, PH.dims(1)) * PH) + matmulTN(Pred, Credi));
							}
						}
					}
					kk++;
					if (kk >= covIter)
						steady = true;
				}
			}
			for (uint64_t tt = 0ULL; tt < Nt - (window - 1ULL); tt++) {
				jg++;
				loadSystemMatrix(storeData, sparseS, tt, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
					sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
				//if (useSmoother) {
				//	KS[tt % N_lag] = Pplus;
				//	if (complexType == 2)
				//		KSi[tt % N_lag] = Pplusi;
				//}
				//mexPrintf("xt.dims(1) = %d\n", xt.dims(1));
				//mexEvalString("pause(.0001);");
				if (complexType == 1 || complexType == 2) {
					xtr = real(xt(span, tt, NN));
					xti = imag(xt(span, tt, NN));
				}
				else if (complexType == 3)
					xtr = join(0, real(xt(span, tt, NN)), imag(xt(span, tt, NN)));
				else
					xtr = xt(span, tt, NN);
				if (DEBUG) {
					mexPrintf("xtr.dims(0) = %d\n", xtr.dims(0));
					mexEvalString("pause(.0001);");
				}
				computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
				if (useSmoother || regularization == 3) {
					cholesky(A, Pplus, false);
					if (complexType == 2)
						cholesky(Ai, Pplusi, false);
					KS[tt % N_lag] = A;
					if (complexType == 2)
						KSi[tt % N_lag] = Ai;
				}
				if (DEBUG) {
					mexPrintf("A.dims(0) = %d\n", A.dims(0));
					mexPrintf("A.dims(1) = %d\n", A.dims(1));
					mexPrintf("A.summa = %f\n", af::sum<float>(flat(A)));
					mexPrintf("Pred.dims(0) = %d\n", Pred.dims(0));
					mexPrintf("Pred.dims(1) = %d\n", Pred.dims(1));
					mexPrintf("Pred.summa = %f\n", af::sum<float>(flat(Pred)));
					mexEvalString("pause(.0001);");
				}
				//mexPrintf("S[0].dims(0) = %d\n", S[0].dims(0));
				//mexPrintf("S[0].dims(1) = %d\n", S[0].dims(1));
				//if (complexS) {
				//	mexPrintf("Si[0].dims(0) = %d\n", Si[0].dims(0));
				//	mexPrintf("Si[0].dims(1) = %d\n", Si[0].dims(1));
				//}
				//mexEvalString("pause(.0001);");
				if (complexType == 3) {
					PH = matmul3(S, Si, Pred, tt % hnU);
				}
				else {
					PH = matmul(S[tt % hnU], Pred);
					if (complexS)
						PHi = matmul(Si[tt % hnU], Pred);
				}
				if (DEBUG) {
					mexPrintf("PH.dims(0) = %d\n", PH.dims(0));
					mexPrintf("PH.dims(1) = %d\n", PH.dims(1));
					mexPrintf("PH.summa = %f\n", af::sum<float>(flat(PH)));
					mexEvalString("pause(.0001);");
				}
				if (useF) {
					if (!useSmoother && regularization != 3)
						//if (useKineticModel) {
						//	B = transpose(solve(A.T(), transpose(Pred), AF_MAT_UPPER));
						//	B(seq(0, B.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], B);
						//}
						//else
						B = transpose(solve(A, transpose(matmul(F[tt % sizeF], Pred)), AF_MAT_LOWER));
					else
						//if (useKineticModel) {
						//	B = matmul(Pred, A);
						//	B(seq(0, B.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], B);
						//}
						//else
						B = matmul(F[tt % sizeF], matmul(Pred, A));
					if (complexType == 2) {
						if (!useSmoother && regularization != 3)
							//if (useKineticModel) {
							//	Bi = transpose(solve(Ai.T(), transpose(Pred), AF_MAT_UPPER));
							//	Bi(seq(0, Bi.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], Bi);
							//}
							//else
							if (complexF)
								Bi = transpose(solve(Ai, transpose(matmul(Fi[tt % sizeF], Pred)), AF_MAT_LOWER));
							else
								Bi = transpose(solve(Ai, transpose(matmul(F[tt % sizeF], Pred)), AF_MAT_LOWER));
						else
							//if (useKineticModel) {
							//	Bi = transpose(solve(Ai.T(), transpose(Pred), AF_MAT_UPPER));
							//	Bi(seq(0, Bi.dims(0) / 2 - 1), span) = matmul(F[kk % sizeF], Bi);
							//}
							//else
							if (complexF)
								Bi = matmul(matmul(Fi[tt % sizeF], Pred), Ai);
							else
								Bi = matmul(matmul(F[tt % sizeF], Pred), Ai);
					}
				}
				else {
					if (!useSmoother && regularization != 3)
						B = transpose(solve(A, Pred.T(), AF_MAT_LOWER));
					else
						B = matmul(Pred, A);
					if (complexType == 2) {
						if (!useSmoother && regularization != 3)
							Bi = transpose(solve(Ai, Pred.T(), AF_MAT_LOWER));
						else
							Bi = matmul(Pred, Ai);
					}
				}
				if (DEBUG) {
					mexPrintf("B.dims(0) = %d\n", B.dims(0));
					mexPrintf("B.dims(1) = %d\n", B.dims(1));
					mexPrintf("B.summa = %f\n", af::sum<float>(flat(B)));
					mexEvalString("pause(.0001);");
				}
				if (sparseQ) {
					Cred = matmulTN(Pred, matmul(Q[tt % sizeQ], Pred)) - matmul(matmulTN(Pred, matmul(Q[tt % sizeQ], B)), solve(matmulTN(B, matmul(Q[tt % sizeQ], B)) + identity(B.dims(1), B.dims(1)), matmulTN(B, matmul(Q[tt % sizeQ], Pred))));
					if (complexType == 2)
						Credi = matmulTN(Pred, matmul(Qi[tt % sizeQ], Pred)) - matmul(matmulTN(Pred, matmul(Qi[tt % sizeQ], Bi)), solve(matmulTN(Bi, matmul(Qi[tt % sizeQ], Bi)) + identity(Bi.dims(1), B.dims(1)), matmulTN(Bi, matmul(Qi[tt % sizeQ], Pred))));
				}
				else {
					Cred = matmulTN(Pred, tile(Q[tt % sizeQ](span, NN), 1, Pred.dims(1)) * Pred) - matmul(matmulTN(Pred, tile(Q[tt % sizeQ](span, NN), 1, B.dims(1)) * B),
						solve(matmulTN(B, tile(Q[tt % sizeQ](span, NN), 1, B.dims(1)) * B) + identity(B.dims(1), B.dims(1)), matmulTN(B, tile(Q[tt % sizeQ](span, NN), 1, Pred.dims(1)) * Pred)));
					//Cred = -matmul(tile(Q[tt % sizeQ](span, NN), 1, B.dims(1)) * B, solve(matmulTN(B, tile(Q[tt % sizeQ](span, NN), 1, B.dims(1)) * B) + identity(B.dims(1), B.dims(1)), B.T() * tile(Q[tt % sizeQ](span, NN).T(), B.dims(1), 1)));
					//mexPrintf("Cred.dims(0) = %d\n", Cred.dims(0));
					//mexPrintf("Cred.dims(1) = %d\n", Cred.dims(1));
					//mexEvalString("pause(.0001);");
					if (complexType == 2) {
						Credi = matmulTN(Pred, tile(Qi[tt % sizeQ](span, NN), 1, Pred.dims(1)) * Pred) - matmul(matmulTN(Pred, tile(Qi[tt % sizeQ](span, NN), 1, Bi.dims(1)) * Bi),
							solve(matmulTN(Bi, tile(Qi[tt % sizeQ](span, NN), 1, Bi.dims(1)) * Bi) + identity(Bi.dims(1), Bi.dims(1)), matmulTN(Bi, tile(Qi[tt % sizeQ](span, NN), 1, Pred.dims(1)) * Pred)));
					}
				}
				if (DEBUG) {
					mexPrintf("Cred.dims(0) = %d\n", Cred.dims(0));
					mexPrintf("Cred.dims(1) = %d\n", Cred.dims(1));
					mexPrintf("Cred.summa = %f\n", af::sum<float>(flat(Cred)));
					mexEvalString("pause(.0001);");
				}
				if (complexType == 3)
					computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
				else
					computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
				if (sparseR) {
					if (!useSmoother && regularization != 3) {
						Pplus = matmulTN(PH, matmul(R[tt % sizeR], PH)) + Cred;
						cholesky(A, Pplus, false);
						alphaEst = solve(A.T(), solve(A, matmulTN(PH, matmul(R[tt % sizeR], SS)), AF_MAT_LOWER), AF_MAT_UPPER);
					}
					else {
						Pplus = inverse(matmulTN(PH, matmul(R[tt % sizeR], PH)) + Cred);
						alphaEst = matmul(Pplus, PH.T(), matmul(R[tt % sizeR], SS));
					}
					xtr = xtr + matmul(Pred, alphaEst);
					if (complexType == 1 || complexType == 2) {
						if (complexS)
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
						else
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
						if (!useSmoother && regularization != 3) {
							if (complexType == 2) {
								if (complexS)
									Pplusi = matmulTN(PHi, matmul(Ri[tt % sizeR], PHi)) + Credi;
								else
									Pplusi = matmulTN(PH, matmul(Ri[tt % sizeR], PH)) + Credi;
								cholesky(Ai, Pplusi, false);
							}
							if (complexType == 2)
								if (complexS)
									alphaEst = solve(Ai.T(), solve(Ai, matmulTN(PHi, matmul(Ri[tt % sizeR], SS)), AF_MAT_LOWER), AF_MAT_UPPER);
								else
									alphaEst = solve(Ai.T(), solve(Ai, matmulTN(PH, matmul(Ri[tt % sizeR], SS)), AF_MAT_LOWER), AF_MAT_UPPER);
							else
								alphaEst = solve(A.T(), solve(A, matmulTN(PH, matmul(R[tt % sizeR], SS)), AF_MAT_LOWER), AF_MAT_UPPER);
						}
						else {
							if (complexType == 2) {
								if (complexS) {
									Pplusi = inverse(matmulTN(PHi, matmul(Ri[tt % sizeR], PHi)) + Credi);
									alphaEst = matmul(Pplusi, PHi.T(), matmul(Ri[tt % sizeR], SS));
								}
								else {
									Pplusi = inverse(matmulTN(PH, matmul(Ri[tt % sizeR], PH)) + Credi);
									alphaEst = matmul(Pplusi, PH.T(), matmul(Ri[tt % sizeR], SS));
								}
							}
							else
								alphaEst = matmul(Pplus, PH.T(), matmul(R[tt % sizeR], SS));
						}
						xti = xti + matmul(Pred, alphaEst);
					}
				}
				else {
					if (!useSmoother && regularization != 3) {
						Pplus = matmul(PH.T(), tile(R[tt % sizeR], 1, PH.dims(1)) * PH) + Cred;
						//Pplus = inverse(matmul(PH.T(), tile(R[tt % sizeR], 1, PH.dims(1)) * PH) + Cred);
						int success = cholesky(A, Pplus, false);
						if (success != 0) {
							mexPrintf("Cholesky failed\n");
							return;
						}
						else if (DEBUG)
							mexPrintf("Cholesky succeeded\n");
						//A = inverse(A);
						//if (complexRef && augType > 0 && regularization == 1)
						//	computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Li, Si, complexType, hnU);
						//else
						alphaEst = solve(A.T(), solve(A, matmulTN(PH, augmentedR(R[tt % sizeR], regularization, RR, xtr.dims(0), complexType) * SS), AF_MAT_LOWER), AF_MAT_UPPER);
						//alphaEst = matmul(solve(A.T(), solve(A, PH.T(), AF_MAT_LOWER), AF_MAT_UPPER), R[tt % sizeR] * (real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))) - matmul(S[tt % hnU], xtr)));
						//alphaEst = matmul(A.T(), A, PH.T(), R[tt % sizeR] * (real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))) - matmul(S[tt % hnU], xtr)));
						//alphaEst = matmul(A, A.T(), PH.T(), R[tt % sizeR] * (real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))) - matmul(S[tt % hnU], xtr)));
					}
					else {
						//Pplus = inverse(matmul(PH.T(), tile(R[tt % sizeR], 1, PH.dims(1)) * PH) + matmulTN(Pred, Cred));
						Pplus = inverse(matmul(PH.T(), tile(R[tt % sizeR], 1, PH.dims(1)) * PH) + Cred);
						//computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
						alphaEst = matmul(Pplus, PH.T(), augmentedR(R[tt % sizeR], regularization, RR, xtr.dims(0), complexType) * SS);
						//alphaEst = matmul(Pplus, PH.T(), R[tt % sizeR] * (real(m0(seq(nMeas * NN + Nm * (tt), nMeas * NN + Nm * (tt + 1) - 1 + Nm * (window - 1ULL)))) - matmul(S[tt % hnU], xtr)));
					}
					//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
					//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
					//mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
					//mexPrintf("A.summa = %f\n", af::sum<float>(flat(A)));
					//mexPrintf("SS.summa = %f\n", af::sum<float>(flat(SS)));
					//mexEvalString("pause(.0001);");
					if (DEBUG) {
						mexPrintf("alphaEst0.dims(0) = %d\n", alphaEst.dims(0));
						mexPrintf("alphaEst0.dims(1) = %d\n", alphaEst.dims(1));
						mexPrintf("alphaEst0.summa = %f\n", af::sum<float>(flat(alphaEst)));
						mexPrintf("A.summa = %f\n", af::sum<float>(flat(A)));
						mexPrintf("Pplus.summa = %f\n", af::sum<float>(flat(Pplus)));
						mexEvalString("pause(.0001);");
					}
					xtr += matmul(Pred, alphaEst);
					if (DEBUG) {
						mexPrintf("xtr.dims(0) = %d\n", xtr.dims(0));
						mexPrintf("xtr.dims(1) = %d\n", xtr.dims(1));
						mexPrintf("xtr.summa = %f\n", af::sum<float>(flat(xtr)));
						mexEvalString("pause(.0001);");
					}
					if (complexType == 1 || complexType == 2) {
						if (complexS)
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
						else
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
						if (complexType == 2) {
							if (!useSmoother && regularization != 3) {
								if (complexS) {
									Pplus = matmul(PHi.T(), tile(Ri[tt % sizeR], 1, PHi.dims(1)) * PHi) + Credi;
									cholesky(Ai, Pplus, false);
									alphaEst = solve(Ai.T(), solve(Ai, matmulTN(PHi, augmentedR(Ri[tt % sizeR], regularization, RR, xti.dims(0), complexType) * SS), AF_MAT_LOWER), AF_MAT_UPPER);
								}
								else {
									Pplus = matmul(PH.T(), tile(Ri[tt % sizeR], 1, PH.dims(1)) * PH) + Credi;
									cholesky(Ai, Pplus, false);
									alphaEst = solve(Ai.T(), solve(Ai, matmulTN(PH, augmentedR(Ri[tt % sizeR], regularization, RR, xti.dims(0), complexType) * SS), AF_MAT_LOWER), AF_MAT_UPPER);
								}
							}
							else {
								if (complexS) {
									Pplusi = inverse(matmul(PHi.T(), tile(Ri[tt % sizeR], 1, PH.dims(1)) * PHi) + Credi);
									alphaEst = matmul(Pplusi, PHi.T(), augmentedR(Ri[tt % sizeR], regularization, RR, xti.dims(0), complexType) * SS);
								}
								else {
									Pplusi = inverse(matmul(PH.T(), tile(Ri[tt % sizeR], 1, PH.dims(1)) * PH) + Credi);
									alphaEst = matmul(Pplusi, PH.T(), augmentedR(Ri[tt % sizeR], regularization, RR, xti.dims(0), complexType) * SS);
								}
							}
						}
						else {
							//computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
							if (!useSmoother && regularization != 3)
								alphaEst = solve(A.T(), solve(A, matmulTN(PH, augmentedR(R[tt % sizeR], regularization, RR, xti.dims(0), complexType) * SS), AF_MAT_LOWER), AF_MAT_UPPER);
							else
								alphaEst = matmul(Pplus, PH.T(), augmentedR(R[tt % sizeR], regularization, RR, xti.dims(0), complexType) * SS);
						}
						if (DEBUG) {
							mexPrintf("alphaEst.dims(0) = %d\n", alphaEst.dims(0));
							mexPrintf("alphaEst.dims(1) = %d\n", alphaEst.dims(1));
							mexPrintf("alphaEst.summa = %f\n", af::sum<float>(flat(alphaEst)));
							mexEvalString("pause(.0001);");
						}
						xti = xti + matmul(Pred, alphaEst);
					}
				}
				if (complexType == 0)
					xt(span, tt + 1, NN) = xtr;
				else if (complexType == 1 || complexType == 2)
					xt(span, tt + 1, NN) = complex(xtr, xti);
				else if (complexType == 3)
					if (kineticModel)
						xt(span, tt + 1, NN) = complex(join(0, xtr(seq(0, imDim - 1)), xtr(seq(imDimU, imDimU + imDim - 1))), join(0, xtr(seq(imDim, imDimU - 1)), xtr(seq(imDimU + imDim, end))));
					else
						xt(span, tt + 1, NN) = complex(xtr(seq(0, xtr.dims(0) / 2 - 1)), xtr(seq(xtr.dims(0) / 2, end)));
				if (DEBUG) {
					mexPrintf("xt.dims(0) = %d\n", xt.dims(0));
					mexPrintf("regularization = %d\n", regularization);
					mexEvalString("pause(.0001);");
				}
				if (regularization > 2) {
					computeDenoising(xt, imDim, tt, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
						huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, Type, Pred);
					if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
						mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
						break;
					}
				}
				if (useSmoother && ((tt == N_lag - 1) || (tt >= N_lag - 1 && jg == skip) || tt == (Nt - 1))) {
					jg = 0;
					int64_t ww = 0;
					int64_t ll = tt % N_lag;
					int64_t jj = tt % N_lag;
					int64_t ff = tt % sizeF;
					int64_t qq = tt % sizeQ;
					int64_t uu = tt % sizeU;
					int64_t gg = tt % sizeG;
					array xp, xpi, Qx, Qxi;
					xlt1(span, end) = xt(seq(0, imDim - 1), tt + 1, NN);
					if (DEBUG) {
						mexPrintf("tt = %d\n", tt);
						mexEvalString("pause(.0001);");
					}
					for (int64_t to = N_lag - 1; to >= 0; to--) {
						//if (to == N_lag - 1) {
						//	//xtp_old = xtp(span, jj);
						//}
						//else {
						//	//xtp_old = xtp(span, jj - ww);
						//}
						//cholesky(A, KS[ll]);
						//if (complexType == 2)
						//	cholesky(Ai, KSi[ll]);
						if (complexType == 3)
							xp = join(0, real(xt(seq(0, imDim - 1), tt - ww, NN)), imag(xt(seq(0, imDim - 1), tt - ww, NN)));
						else
							xp = real(xt(seq(0, imDim - 1), tt - ww, NN));
						if (complexType == 1 || complexType == 2)
							xpi = imag(xt(seq(0, imDim - 1), tt - ww, NN));
						computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xp, xpi, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
						if (complexType == 3)
							xp = join(0, real(xlt1(span, to + 1)), imag(xlt1(span, to + 1))) - xp;
						else
							xp = real(xlt1(span, to + 1)) - xp;
						if (complexType == 2 || complexType == 1)
							xpi = imag(xlt1(span, to + 1)) - xpi;
						//mexPrintf("xp.summa = %f\n", af::sum<float>(flat(xp)));
						//mexEvalString("pause(.0001);");
						if (sparseQ) {
							Qx = matmul(Q[qq], xp);
							if (complexType == 2)
								Qxi = matmul(Qi[qq], xpi);
							else if (complexType == 1)
								Qxi = matmul(Q[qq], xpi);
						}
						else {
							Qx = Q[qq](span, NN) * xp;
							if (complexType == 2)
								Qxi = Qi[qq](span, NN) * xpi;
							else if (complexType == 1)
								Qxi = Q[qq](span, NN) * xpi;
						}
						//mexPrintf("Qx.dims(0) = %d\n", Qx.dims(0));
						//mexPrintf("Qx.dims(1) = %d\n", Qx.dims(1));
						//mexEvalString("pause(.0001);");
						if (useF) {
							//if (useKineticModel) {
							//	B = matmul(Pred, A);
							//	B(seq(0, B.dims(0) / 2 - 1), span) = matmul(F[ff], B);
							//	if (complexType == 2) {
							//		Bi = matmul(Pred, Ai);
							//		Bi(seq(0, Bi.dims(0) / 2 - 1), span) = matmul(F[ff], Bi);
							//	}
							//}
							//else {
							B = matmul(matmul(F[ff], Pred), KS[jj]);
							if (complexType == 2) {
								if (complexF)
									Bi = matmul(matmul(Fi[ff], Pred), KSi[jj]);
								else
									Bi = matmul(matmul(F[ff], Pred), KSi[jj]);
							}
							//}
						}
						else {
							B = matmul(Pred, KS[jj]);
							if (complexType == 2) {
								Bi = matmul(Pred, KSi[jj]);
							}
						}
						if (sparseQ) {
							Cred = Qx - matmul(matmul(Q[tt % sizeQ], B), solve(matmulTN(B, matmul(Q[tt % sizeQ], B)) + identity(B.dims(1), B.dims(1)), matmulTN(B, Qx)));
							if (complexType == 2)
								Credi = Qxi - matmul(matmul(Qi[tt % sizeQ], Bi), solve(matmulTN(Bi, matmul(Qi[tt % sizeQ], Bi)) + identity(Bi.dims(1), Bi.dims(1)), matmulTN(Bi, Qxi)));
						}
						else {
							Cred = Qx - matmul(tile(Q[tt % sizeQ](span, NN), 1, B.dims(1)) * B, solve(matmulTN(B, tile(Q[tt % sizeQ](span, NN), 1, B.dims(1)) * B) + identity(B.dims(1), B.dims(1)), matmulTN(B, Qx)));
							if (complexType == 2)
								Credi = Qxi - matmul(tile(Q[tt % sizeQ](span, NN), 1, Bi.dims(1)) * Bi, solve(matmulTN(Bi, tile(Qi[tt % sizeQ](span, NN), 1, Bi.dims(1)) * Bi) + identity(Bi.dims(1), Bi.dims(1)), matmulTN(Bi, Qxi)));
						}
						//mexPrintf("Cred.dims(0) = %d\n", Cred.dims(0));
						//mexPrintf("Cred.dims(1) = %d\n", Cred.dims(1));
						//mexPrintf("Cred.summa = %f\n", af::sum<float>(flat(Cred)));
						//mexEvalString("pause(.0001);");
						if (useF) {
							//if (useKineticModel) {
							//	Cred(seq(0, Cred.dims(0) / 2 - 1), span) = matmul(F[ff], Cred);
							//	if (complexType == 2)
							//		Credi(seq(0, Credi.dims(0) / 2 - 1), span) = matmul(F[ff], Credi);
							//}
							//else {
							Cred = matmul(F[ff], Cred);
							if (complexType == 2)
								if (complexF)
									Credi = matmul(Fi[ff], Credi);
								else
									Credi = matmul(F[ff], Credi);
							//}
						}
						Cred = matmul(Pred, KS[jj], matmul(KS[jj].T(), Pred.T(), Cred));
						//mexPrintf("Cred.dims(0) = %d\n", Cred.dims(0));
						//mexPrintf("Cred.dims(1) = %d\n", Cred.dims(1));
						//mexPrintf("Cred.summa = %f\n", af::sum<float>(flat(Cred)));
						//mexEvalString("pause(.0001);");
						if (complexType == 2)
							Credi = matmul(Pred, KSi[jj], matmul(KSi[jj].T(), Pred.T(), Credi));
						if (complexType == 0)
							xlt1(span, to) = xt(seq(0, imDim - 1), tt - ww) + Cred;
						else if (complexType == 1)
							xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), tt - ww, NN)) + Cred, imag(xt(seq(0, imDim - 1), tt - ww, NN)) + Cred);
						else if (complexType == 2)
							xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), tt - ww, NN)) + Cred, imag(xt(seq(0, imDim - 1), tt - ww, NN)) + Credi);
						else if (complexType == 3)
							xlt1(span, to) = join(0, real(xt(seq(0, imDim - 1), tt - ww, NN)), imag(xt(seq(0, imDim - 1), tt - ww, NN))) + Cred;
						ww++;
						//mexPrintf("ww = %d\n", ww);
						ll++;
						ff--;
						jj--;
						gg--;
						uu--;
						qq--;
						if (ll == N_lag)
							ll = 0;
						if (ff < 0)
							ff = sizeF - 1;
						if (jj < 0)
							jj = N_lag - 1;
						if (qq < 0)
							qq = sizeQ - 1;
						if (uu < 0)
							uu = sizeU - 1;
						if (gg < 0)
							gg = sizeG - 1;
					}
					if (complexType == 3)
						xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = complex(xlt1(seq(0, xlt1.dims(0) / 2 - 1), seq(0, end - 1LL)), xlt1(seq(xlt1.dims(0) / 2, end), seq(0, end - 1LL)));
					else
						xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = xlt1(span, seq(0, N_lag - 1LL));
				}
			}
		}
		else if (algorithm == 10) {

			Type = 3;
			array a0, a0i, r0, p0, E, temp2, vhat, vhati, v0, U, Ss, VV;


			if (useSmoother) {
				if (complexType == 0 || complexType == 3)
					xlt1 = constant(0.f, imDimU, N_lag + 1);
				else
					xlt1 = constant(0.f, imDimU, N_lag + 1, c32);
				KS.resize(N_lag);
				if (complexType == 2)
					KSi.resize(N_lag);
			}
			//vhat = constant(1.f, imDimN, 1);
			//xtr = real(xt(span, 0, NN));
			for (uint64_t tt = 0ULL; tt < Nt - (window - 1ULL); tt++) {

				loadSystemMatrix(storeData, sparseS, tt, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
					sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
				if (complexType == 3)
					xtr = join(0, real(xt(span, tt, NN)), imag(xt(span, tt, NN)));
				else
					xtr = real(xt(span, tt, NN));
				if (complexType == 1 || complexType == 2)
					xti = imag(xt(span, tt, NN));
				computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
				if (complexType <= 2)
					computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
				else
					computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);

				if (useSmoother) {
					KS[tt % N_lag] = Pplus;
					if (complexType == 2)
						KSi[tt % N_lag] = Pplusi;
				}

				//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
				//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
				////mexPrintf("Q.summa = %f\n", af::sum<float>(flat(Q[tt % sizeQ](span, NN))));
				//mexEvalString("pause(.0001);");
				if (tt <= 2ULL) {
					//	a0 = SS;
					setSeed(0);
					vhat = randn(imDim3, 1);
					setSeed(0);
					a0 = randn(NmU, 1);
				}
				array temp;
				if (complexType == 3)
					temp = vecmul3(S, Si, a0, tt % hnU, true);
				else
					temp = matmul(S[tt % hnU], a0, AF_MAT_TRANS);
				//mexPrintf("temp.dims(0) = %d\n", temp.dims(0));
				//mexPrintf("temp.dims(1) = %d\n", temp.dims(1));
				//mexEvalString("pause(.0001);");
				if (sparseQ)
					temp2 = matmul(Q[tt % sizeQ], temp);
				else
					temp2 = temp * Q[tt % sizeQ](span, NN);
				if (complexType == 3)
					temp2 = vecmul3(S, Si, temp2, tt % hnU, false);
				else
					temp2 = matmul(S[tt % hnU], temp2);
				//mexPrintf("temp2.dims(0) = %d\n", temp2.dims(0));
				//mexPrintf("temp2.dims(1) = %d\n", temp2.dims(1));
				//mexEvalString("pause(.0001);");
				computePminusCG(useF, sparseF, useKineticModel, 0, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
				if (complexType == 3)
					temp = vecmul3(S, Si, temp, tt % hnU, false);
				else
					temp = matmul(S[tt % hnU], temp);
				//mexPrintf("temp.dims(0) = %d\n", temp.dims(0));
				//mexPrintf("temp.dims(1) = %d\n", temp.dims(1));
				//mexEvalString("pause(.0001);");
				if (sparseR)
					r0 = matmul(R[tt % sizeR], a0);
				else
					r0 = a0 * R[tt % sizeR];
				r0 = SS - (temp + temp2 + r0);
				//mexPrintf("r0.dims(0) = %d\n", r0.dims(0));
				//mexPrintf("r0.dims(1) = %d\n", r0.dims(1));
				//mexEvalString("pause(.0001);");
				p0 = r0;
				array V = constant(0.f, NmU, cgIter);
				array P = constant(0.f, NmU, cgIter);
				array PP = constant(0.f, NmU, cgIter);
				array D = constant(0.f, cgIter, 1);

				for (uint32_t ii = 0; ii < cgIter; ii++) {
					P(span, ii) = p0;
					V(span, ii) = r0;
					if (complexType == 3)
						temp = vecmul3(S, Si, p0, tt % hnU, true);
					else
						temp = matmul(S[tt % hnU], p0, AF_MAT_TRANS);
					//mexPrintf("temp.dims(0) = %d\n", temp.dims(0));
					//mexPrintf("temp.dims(1) = %d\n", temp.dims(1));
					//mexEvalString("pause(.0001);");
					if (sparseQ)
						temp2 = matmul(Q[tt % sizeQ], temp);
					else
						temp2 = temp * Q[tt % sizeQ](span, NN);
					if (complexType == 3)
						temp2 = vecmul3(S, Si, temp2, tt % hnU, false);
					else
						temp2 = matmul(S[tt % hnU], temp2);
					computePminusCG(useF, sparseF, useKineticModel, 0, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
					if (complexType == 3)
						temp = vecmul3(S, Si, temp, tt % hnU, false);
					else
						temp = matmul(S[tt % hnU], temp);
					array ra;
					if (sparseR)
						ra = matmul(R[tt % sizeR], p0);
					else
						ra = p0 * R[tt % sizeR];
					const array rk = temp + temp2 + ra;
					PP(span, ii) = rk;
					D(ii) = 1.f / sqrt(dot(p0, rk));
					const array gam = dot(r0, r0) / dot(p0, rk);
					a0 = a0 + tile(gam, p0.dims(0), 1) * p0;
					ra = r0 - tile(gam, rk.dims(0), 1) * rk;
					ra = MGSOG(join(1, V(span, seq(0, ii)), ra));
					ra = ra(span, end);
					const array be = -dot(ra, ra) / dot(r0, r0);
					p0 = ra - tile(be, p0.dims(0), 1) * p0;
					if (forceOrthogonalization) {
						array pa = MGSOG(join(1, PP(span, seq(0, ii)), p0));
						p0 = pa(span, end);
					}
					r0 = ra;
					//mexPrintf("norm(r0) = %f\n", norm(r0));
					//mexEvalString("pause(.0001);");
					if (norm(r0) < cgThreshold)
						break;
				}
				if (computeConsistency && tt >= initialSteps) {
					storeConsistency(tt, Nt, stepSize, eps, SS, a0, cc, v, sparseR, regularization, Pplus, epsP, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, false, 1);
				}

				if (complexType == 3)
					temp = vecmul3(S, Si, a0, tt % hnU, true);
				else
					temp = matmul(S[tt % hnU], a0, AF_MAT_TRANS);
				//mexPrintf("temp.dims(0) = %d\n", temp.dims(0));
				//mexPrintf("temp.dims(1) = %d\n", temp.dims(1));
				//mexEvalString("pause(.0001);");
				if (sparseQ)
					temp2 = matmul(Q[tt % sizeQ], temp);
				else
					temp2 = temp * Q[tt % sizeQ](span, NN);
				computePminusCG(useF, sparseF, useKineticModel, 0, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
				xtr += temp + temp2;

				E = P * tile(D.T(), P.dims(0), 1);
				V = constant(0.f, imDim3, cgIter);
				array VA = constant(0.f, imDim3, cgIter);
				P = constant(0.f, imDim3, cgIter);
				D = constant(0.f, cgIter - 1, 1);
				array alpha = constant(0.f, cgIter, 1);

				v0 = vhat / tile(sqrt(sum(vhat * vhat)), vhat.dims(0), 1);
				//mexPrintf("vhat.dims(0) = %d\n", vhat.dims(0));
				//mexPrintf("vhat.dims(1) = %d\n", vhat.dims(1));
				//mexEvalString("pause(.0001);");
				temp = v0;
				computePminusCG(useF, sparseF, useKineticModel, 0, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
				if (sparseQ)
					temp += matmul(Q[tt % sizeQ], v0);
				else
					temp += v0 * Q[tt % sizeQ](span, NN);
				//mexPrintf("temp1.dims(0) = %d\n", temp.dims(0));
				//mexPrintf("temp1.dims(1) = %d\n", temp.dims(1));
				//mexEvalString("pause(.0001);");
				if (complexType == 3)
					temp2 = vecmul3(S, Si, matmul(E, E.T(), vecmul3(S, Si, temp, tt % hnU, false)), tt % hnU, true);
				else
					temp2 = matmul(S[tt % hnU], matmul(E, E.T(), matmul(S[tt % hnU], temp)), AF_MAT_TRANS);
				//mexPrintf("temp2.dims(0) = %d\n", temp2.dims(0));
				//mexPrintf("temp2.dims(1) = %d\n", temp2.dims(1));
				//mexEvalString("pause(.0001);");
				if (sparseQ)
					temp -= matmul(Q[tt % sizeQ], temp2);
				else
					temp -= temp2 * Q[tt % sizeQ](span, NN);
				computePminusCG(useF, sparseF, useKineticModel, 0, temp2, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
				temp2 = temp - temp2;
				alpha(0) = dot(v0, temp2);
				vhat = temp2 - v0 * tile(alpha(0), v0.dims(0), 1);
				V(span, 0) = v0;
				VA(span, 0) = vhat;
				int64_t iu = 0LL;
				for (int64_t ii = 1; ii < cgIter; ii++) {
					iu++;
					D(ii - 1LL) = sqrt(sum(vhat * vhat));
					V(span, ii) = vhat / tile(D(ii - 1), vhat.dims(0), 1);
					temp = V(span, ii);
					computePminusCG(useF, sparseF, useKineticModel, 0, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
					if (sparseQ)
						temp += matmul(Q[tt % sizeQ], V(span, ii));
					else
						temp += V(span, ii) * Q[tt % sizeQ](span, NN);
					if (complexType == 3)
						temp2 = vecmul3(S, Si, matmul(E, E.T(), vecmul3(S, Si, temp, tt % hnU, false)), tt % hnU, true);
					else
						temp2 = matmul(S[tt % hnU], matmul(E, E.T(), matmul(S[tt % hnU], temp)), AF_MAT_TRANS);
					if (sparseQ)
						temp -= matmul(Q[tt % sizeQ], temp2);
					else
						temp -= temp2 * Q[tt % sizeQ](span, NN);
					computePminusCG(useF, sparseF, useKineticModel, 0, temp2, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
					temp2 = temp - temp2;
					const array u = temp2 - tile(D(ii - 1), V(span, ii - 1).dims(0), 1) * V(span, ii - 1);
					alpha(ii) = dot(V(span, ii), u);
					vhat = u - tile(alpha(ii), V(span, ii).dims(0), 1) * V(span, ii);
					if (forceOrthogonalization) {
						vhat = MGSOG(join(1, VA(span, seq(0, ii)), vhat));
						vhat = vhat(span, end);
						VA(span, ii) = vhat;
					}
					//mexPrintf("norm(vhat) = %f\n", norm(vhat));
					//mexEvalString("pause(.0001);");
					if (norm(vhat) < cgThreshold)
						break;
				}
				array T = constant(0.f, iu + 1, iu + 1);
				T(seq(0, end, iu + 2LL)) = alpha(seq(0, iu));
				T(seq(1, end, iu + 2LL)) = D(seq(0, iu - 1LL));
				T(seq(iu + 2LL, end, iu + 2LL)) = D(seq(0, iu - 1LL));
				af::svd(U, Ss, VV, T);
				Pplus = matmul(V(span, seq(0, iu)), U) * tile(sqrt(Ss).T(), V.dims(0), 1);
				if (computeConsistency && tt >= initialSteps && computeBayesianP) {
					computeInnovationCov(HH, sparseR, R, Pplus, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, false, 1);
					if (complexType == 3)
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
					else
						computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
					storeConsistency(tt, Nt, stepSize, eps, SS, HH, cc, v, sparseR, regularization, Pplus, epsP, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, true);
				}
				if (complexType == 0)
					xt(span, tt + 1, NN) = xtr;
				else if (complexType == 1 || complexType == 2) {
					if (complexS)
						if (complexRef && augType > 0)
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
					else
						if (complexRef && augType > 0)
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);

					if (tt <= 2ULL) {
						//	a0 = SS;
						//setSeed(0);
						vhati = randn(imDim3, 1);
						//setSeed(0);
						a0i = randn(Nm, 1);
					}

					array temp;
					if (complexS)
						temp = matmul(Si[tt % hnU], a0i, AF_MAT_TRANS);
					else
						temp = matmul(S[tt % hnU], a0i, AF_MAT_TRANS);
					if (complexType == 2)
						if (sparseQ)
							temp2 = matmul(Qi[tt % sizeQ], temp);
						else
							temp2 = temp * Qi[tt % sizeQ](span, NN);
					else
						if (sparseQ)
							temp2 = matmul(Q[tt % sizeQ], temp);
						else
							temp2 = temp * Q[tt % sizeQ](span, NN);
					if (complexS)
						temp2 = matmul(Si[tt % hnU], temp2);
					else
						temp2 = matmul(S[tt % hnU], temp2);
					computePminusCG(useF, sparseF, useKineticModel, complexType, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
					if (complexS)
						temp = matmul(Si[tt % hnU], temp);
					else
						temp = matmul(S[tt % hnU], temp);
					if (complexType == 2)
						if (sparseR)
							r0 = matmul(Ri[tt % sizeQ], a0i);
						else
							r0 = a0i * Ri[tt % sizeQ];
					else
						if (sparseR)
							r0 = matmul(R[tt % sizeQ], a0i);
						else
							r0 = a0i * R[tt % sizeQ];
					r0 = SS - (temp + temp2 + r0);
					p0 = r0;
					array V = constant(0.f, Nm, cgIter);
					array P = constant(0.f, Nm, cgIter);
					array D = constant(0.f, cgIter, 1);
					array PP = constant(0.f, Nm, cgIter);

					for (uint32_t ii = 0; ii < cgIter; ii++) {
						P(span, ii) = p0;
						V(span, ii) = r0;
						if (complexS)
							temp = matmul(Si[tt % hnU], p0, AF_MAT_TRANS);
						else
							temp = matmul(S[tt % hnU], p0, AF_MAT_TRANS);
						if (complexType == 2)
							if (sparseQ)
								temp2 = matmul(Qi[tt % sizeQ], temp);
							else
								temp2 = temp * Qi[tt % sizeQ](span, NN);
						else
							if (sparseQ)
								temp2 = matmul(Q[tt % sizeQ], temp);
							else
								temp2 = temp * Q[tt % sizeQ](span, NN);
						if (complexS)
							temp2 = matmul(Si[tt % hnU], temp2);
						else
							temp2 = matmul(S[tt % hnU], temp2);
						computePminusCG(useF, sparseF, useKineticModel, complexType, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
						if (complexS)
							temp = matmul(Si[tt % hnU], temp);
						else
							temp = matmul(S[tt % hnU], temp);
						array ra;
						if (complexType == 2)
							if (sparseR)
								ra = matmul(Ri[tt % sizeQ], p0);
							else
								ra = p0 * Ri[tt % sizeQ];
						else
							if (sparseR)
								ra = matmul(R[tt % sizeQ], p0);
							else
								ra = p0 * R[tt % sizeQ];
						const array rk = temp + temp2 + ra;
						PP(span, ii) = rk;
						D(ii) = 1.f / sqrt(dot(p0, rk));
						const array gam = dot(r0, r0) / dot(p0, rk);
						a0i += tile(gam, p0.dims(0), 1) * p0;
						ra = r0 - tile(gam, rk.dims(0), 1) * rk;
						ra = MGSOG(join(1, V(span, seq(0, ii)), ra));
						ra = ra(span, end);
						const array be = -dot(ra, ra) / dot(r0, r0);
						p0 = ra - tile(be, p0.dims(0), 1) * p0;
						if (forceOrthogonalization) {
							array pa = MGSOG(join(1, PP(span, seq(0, ii)), p0));
							p0 = pa(span, end);
						}
						r0 = ra;
						if (norm(r0) < cgThreshold)
							break;
					}

					if (computeConsistency && tt >= initialSteps) {
						storeConsistency(tt, Nt, stepSize, epsi, SS, a0i, cc, vvi, sparseR, regularization, Pplusi, epsPi, computeBayesianP, Ri[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, 1);
					}

					if (complexS)
						temp = matmul(Si[tt % hnU], a0i, AF_MAT_TRANS);
					else
						temp = matmul(S[tt % hnU], a0i, AF_MAT_TRANS);
					if (complexType == 2) {
						if (sparseQ)
							temp2 = matmul(Qi[tt % sizeQ], temp);
						else
							temp2 = temp * Qi[tt % sizeQ](span, NN);
					}
					else {
						if (sparseQ)
							temp2 = matmul(Q[tt % sizeQ], temp);
						else
							temp2 = temp * Q[tt % sizeQ](span, NN);
					}
					computePminusCG(useF, sparseF, useKineticModel, complexType, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
					xti += temp + temp2;
					if (complexType == 2) {
						E = P * tile(D.T(), P.dims(0), 1);
						V = constant(0.f, imDim3, cgIter);
						array VA = constant(0.f, imDim3, cgIter);
						P = constant(0.f, imDim3, cgIter);
						D = constant(0.f, cgIter - 1, 1);
						array alpha = constant(0.f, cgIter, 1);

						v0 = vhat / tile(sqrt(sum(vhat * vhat)), vhat.dims(0), 1);
						temp = v0;
						computePminusCG(useF, sparseF, useKineticModel, complexType, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
						if (sparseQ)
							temp += matmul(Qi[tt % sizeQ], v0);
						else
							temp += v0 * Qi[tt % sizeQ](span, NN);
						if (complexS)
							temp2 = matmul(Si[tt % hnU], matmul(E, E.T(), matmul(Si[tt % hnU], temp)), AF_MAT_TRANS);
						else
							temp2 = matmul(S[tt % hnU], matmul(E, E.T(), matmul(S[tt % hnU], temp)), AF_MAT_TRANS);
						if (sparseQ)
							temp -= matmul(Qi[tt % sizeQ], temp2);
						else
							temp -= temp2 * Qi[tt % sizeQ](span, NN);
						computePminusCG(useF, sparseF, useKineticModel, complexType, temp2, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
						temp2 = temp - temp2;
						alpha(0) = dot(v0, temp2);
						vhati = temp2 - v0 * tile(alpha(0), v0.dims(0), 1);
						V(span, 0) = v0;
						VA(span, 0) = vhat;
						int64_t iu = 0LL;
						for (int64_t ii = 1; ii < cgIter; ii++) {
							iu++;
							D(ii - 1) = sqrt(sum(vhati * vhati));
							V(span, ii) = vhati / tile(D(ii - 1), vhati.dims(0), 1);
							temp = V(span, ii);
							computePminusCG(useF, sparseF, useKineticModel, complexType, temp, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
							if (sparseQ)
								temp += matmul(Qi[tt % sizeQ], V(span, ii));
							else
								temp += V(span, ii) * Qi[tt % sizeQ](span, NN);
							if (complexS)
								temp2 = matmul(Si[tt % hnU], matmul(E, E.T(), matmul(Si[tt % hnU], temp)), AF_MAT_TRANS);
							else
								temp2 = matmul(S[tt % hnU], matmul(E, E.T(), matmul(S[tt % hnU], temp)), AF_MAT_TRANS);
							if (sparseQ)
								temp -= matmul(Qi[tt % sizeQ], temp2);
							else
								temp -= temp2 * Qi[tt % sizeQ](span, NN);
							computePminusCG(useF, sparseF, useKineticModel, complexType, temp2, F[tt % sizeF], Pplus, Pplusi, complexF, Fi[tt % sizeF]);
							temp2 = temp - temp2;
							const array u = temp2 - tile(D(ii - 1), V(span, ii - 1).dims(0), 1) * V(span, ii - 1);
							alpha(ii) = dot(V(span, ii), u);
							vhati = u - tile(alpha(ii), V(span, ii).dims(0), 1) * V(span, ii);
							if (forceOrthogonalization) {
								vhat = MGSOG(join(1, VA(span, seq(0, ii)), vhat));
								vhat = vhat(span, end);
								VA(span, ii) = vhat;
							}
							if (norm(vhat) < cgThreshold)
								break;
						}
						array T = constant(0.f, iu + 1, iu + 1);
						T(seq(0, end, iu + 2LL)) = alpha(seq(0, iu));
						T(seq(1, end, iu + 2LL)) = D(seq(0, iu - 1LL));
						T(seq(iu + 2LL, end, iu + 2LL)) = D(seq(0, iu - 1LL));
						af::svd(U, Ss, VV, T);
						Ss = sqrt(Ss);
						Pplusi = matmul(V(span, seq(0, iu)), U) * tile(sqrt(Ss).T(), V.dims(0), 1);
					}
					if (computeConsistency && tt >= initialSteps && computeBayesianP) {
						if (complexType == 2)
							computeInnovationCov(HHi, sparseR, Ri, Pplusi, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, complexS, 1);
						else
							computeInnovationCov(HH, sparseR, R, Pplus, S, tt, hnU, sizeR, tt % hnU, RR, regularization, complexType, Si, 1);
						if (complexS)
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
						else
							if (complexRef && augType > 0 && regularization == 1)
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
							else
								computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
						if (complexType == 2)
							if (complexS)
								storeConsistency(tt, Nt, stepSize, epsi, SS, HHi, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, Si[tt % hnU], complexType, Si, hnU, true);
							else
								storeConsistency(tt, Nt, stepSize, epsi, SS, HHi, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, true);
						else
							storeConsistency(tt, Nt, stepSize, epsi, SS, HH, cc, vvi, sparseR, regularization, Pplus, epsPi, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, true);
					}
					xt(span, tt + 1, NN) = complex(xtr, xti);
				}
				else if (complexType == 3)
					if (kineticModel)
						xt(span, tt + 1, NN) = complex(join(0, xtr(seq(0, imDim - 1)), xtr(seq(imDimU, imDimU + imDim - 1))), join(0, xtr(seq(imDim, imDimU - 1)), xtr(seq(imDimU + imDim, end))));
					else
						xt(span, tt + 1, NN) = complex(xtr(seq(0, xtr.dims(0) / 2 - 1)), xtr(seq(xtr.dims(0) / 2, end)));
				if (useSmoother && ((tt == N_lag - 1) || (tt >= N_lag - 1 && jg == skip) || tt == (Nt - 1))) {
					jg = 0;
					int64_t ww = 0;
					int64_t ll = tt % N_lag;
					int64_t jj = tt % N_lag;
					int64_t ji = tt % N_lag;
					if (complexType != 2)
						ji = 0;
					int64_t qq = tt % sizeQ;
					int64_t q2 = qq;
					int64_t ff = tt % sizeF;
					if (complexType == 3)
						xlt1(span, end) = join(0, real(xt(seq(0, imDim - 1), tt + 1, NN)), imag(xt(seq(0, imDim - 1), tt + 1, NN)));
					else
						xlt1(span, end) = xt(seq(0, imDim - 1), tt + 1, NN);
					array xTemp, xTempi, Pqi, Pq;
					mexPrintf("N_lag = %d\n", N_lag);
					mexPrintf("tt = %d\n", tt);
					mexPrintf("xlt1.dims(0) = %d\n", xlt1.dims(0));
					mexPrintf("xlt1.dims(1) = %d\n", xlt1.dims(1));
					mexEvalString("pause(.0001);");
					for (int64_t to = N_lag - 1; to >= 0; to--) {
						if (complexType == 3)
							xtr = join(0, real(xt(span, tt - ww, NN)), imag(xt(span, tt - ww, NN)));
						else
							xtr = real(xt(span, tt - ww, NN));
						if (complexType == 1 || complexType == 2)
							xti = imag(xt(span, tt - ww, NN));
						if (useF || useG || useU)
							computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
						xTemp = real(xlt1(span, to + 1)) - xtr;
						if (complexType == 2 || complexType == 1)
							xTempi = imag(xlt1(span, to + 1)) - xti;
						//Pq = xTemp;
						//if (complexType == 2 || complexType == 1)
						//	Pqi = xTempi;
						if (!sparseQ) {
							Q2[0] = 1.f / Q[qq](span, NN);
							if (complexType == 2)
								Q2i[0] = 1.f / Qi[qq](span, NN);
							q2 = 0;
						}
						else
							q2 = qq;
						mexPrintf("xTemp.dims(0) = %d\n", xTemp.dims(0));
						mexPrintf("xTemp.dims(1) = %d\n", xTemp.dims(1));
						mexPrintf("xTemp.summa = %f\n", af::sum<float>(flat(xTemp)));
						mexEvalString("pause(.0001);");
						Pq = matrixInversionP(sparseQ, useF, complexF, complexType, xTemp, xTempi, Q2, Q2i, F, Fi, NN, q2, ff, KS[jj], KSi[ji]);
						if (complexType == 2 || complexType == 1) {
							Pqi = Pq(span, end);
							Pq = Pq(span, 0);
						}
						mexPrintf("Pq3.dims(0) = %d\n", Pq.dims(0));
						mexPrintf("Pq3.dims(1) = %d\n", Pq.dims(1));
						mexPrintf("Pq.summa = %f\n", af::sum<float>(flat(Pq)));
						mexEvalString("pause(.0001);");
						if (useF) {
							Pq = matmul(F[ff], Pq, AF_MAT_TRANS);
							if (complexType == 2)
								if (complexF)
									Pqi = matmul(Fi[ff], Pqi, AF_MAT_TRANS);
								else
									Pqi = matmul(F[ff], Pqi, AF_MAT_TRANS);
							else if (complexType == 1)
								Pqi = matmul(F[ff], Pqi, AF_MAT_TRANS);
						}
						//array testi = matmulNT(KS[jj], KS[jj]);
						//array testi2 = diag(testi, 0, true);
						//array Pm = diag(Q2[0], 0, false) - tile(Q2[0], 1, imDim) * matmul(KS[jj], solve(identity(KS[jj].dims(1), KS[jj].dims(1)) + matmulTN(KS[jj], tile(Q2[0], 1, KS[jj].dims(1)) * KS[jj]), KS[jj].T() * tile(Q2[0].T(), KS[jj].dims(1), 1)));
						//array Pm2 = matmulNT(KS[jj], KS[jj]) + diag(Q[qq](span, NN), 0, false);
						//array testi3 = solve(Pm2, testi);
						//array testi4 = diag(Pm, 0, true);
						//array testi5 = diag(Pm2, 0, true);
						//array testi6 = diag(testi3, 0, true);
						Pq = matmul(KS[jj], KS[jj].T(), Pq);
						//mexPrintf("Pq4.dims(0) = %d\n", Pq.dims(0));
						//mexPrintf("Pq4.dims(1) = %d\n", Pq.dims(1));
						//mexPrintf("Pq.summa = %f\n", af::sum<float>(flat(Pq)));
						//mexPrintf("testi.summa = %f\n", af::sum<float>(flat(testi)));
						//mexPrintf("testi2.summa = %f\n", af::sum<float>(flat(testi2)));
						//mexPrintf("Pm.summa = %f\n", af::sum<float>(flat(Pm)));
						//mexPrintf("Pm2.summa = %f\n", af::sum<float>(flat(Pm2)));
						//mexPrintf("testi3.summa = %f\n", af::sum<float>(flat(testi3)));
						//mexPrintf("testi4.summa = %f\n", af::sum<float>(flat(testi4)));
						//mexPrintf("testi5.summa = %f\n", af::sum<float>(flat(testi5)));
						//mexPrintf("testi6.summa = %f\n", af::sum<float>(flat(testi6)));
						//mexEvalString("pause(.0001);");
						if (complexType == 2)
							Pqi = matmul(KSi[ji], KSi[ji].T(), Pqi);
						else if (complexType == 1)
							Pqi = matmul(KS[jj], KS[jj].T(), Pqi);
						if (complexType == 0)
							xlt1(span, to) = xt(seq(0, imDim - 1), tt - ww) + Pq;
						else if (complexType == 1)
							xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), tt - ww, NN)) + Pq, imag(xt(seq(0, imDim - 1), tt - ww, NN)) + Pqi);
						else if (complexType == 2)
							xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), tt - ww, NN)) + Pq, imag(xt(seq(0, imDim - 1), tt - ww, NN)) + Pqi);
						else if (complexType == 3)
							xlt1(span, to) = join(0, real(xt(seq(0, imDim - 1), tt - ww, NN)), imag(xt(seq(0, imDim - 1), tt - ww, NN))) + Pq;
						mexPrintf("xlt1(span, to + 1).summa = %f\n", af::sum<float>(flat(real(xlt1(span, to + 1)))));
						//mexPrintf("xlt1(span, to + 1).summa = %f\n", af::sum<float>(flat(real(xt(seq(0, imDim - 1), tt - ww + 1, NN)))));
						mexPrintf("ww = %d\n", ww);
						ww++;
						jj--;
						ji--;
						ll++;
						qq--;
						ff--;
						if (jj < 0)
							jj = N_lag - 1;
						if (ji < 0)
							if (complexType == 2)
								ji = N_lag - 1;
							else
								ji = 0;
						if (ll == N_lag)
							ll = 0;
						if (ff < 0)
							ff = 0LL;
						if (qq < 0)
							qq = 0LL;
					}
					if (complexType == 3)
						xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = complex(xlt1(seq(0, xlt1.dims(0) / 2 - 1), seq(0, end - 1LL)), xlt1(seq(xlt1.dims(0) / 2, end), seq(0, end - 1LL)));
					else
						xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = xlt1(span, seq(0, N_lag - 1LL));
				}
				if (regularization > 2) {
					computeDenoising(xt, imDim, tt, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
						huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, Type, Pred);
					if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
						mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
						break;
					}
				}
			}
			if (storeCovariance == 2)
				if (complexType == 2)
					if (kineticModel)
						P1(span, seq(0, Pplus.dims(1) - 1)) = complex(Pplus(seq(0, imDim - 1), span), Pplusi(seq(0, imDim - 1), span));
					else
						P1(span, seq(0, Pplus.dims(1) - 1)) = complex(Pplus, Pplusi);
				else if (complexType == 3)
					P1(span, seq(0, Pplus.dims(1) - 1)) = complex(Pplus(seq(0, imDim - 1), span), Pplus(seq(imDim, end), span));
				else
					if (kineticModel)
						P1(span, seq(0, Pplus.dims(1) - 1)) = Pplus(seq(0, imDimU - 1), span);
					else
						P1(span, seq(0, Pplus.dims(1) - 1)) = Pplus;
			if (computeConsistency) {
				computeConsistencyTests(cc, Nt, stepSize, initialSteps, v, Nm, eps, epsP, computeBayesianP, complexType, vvi, epsi, epsPi);
			}
		}
		else if (algorithm == 11) {
			Type = 3;
			array a0, a0i, r0, p0, E, temp2, tempi, temp, vhat, vhati, v0, U, Ss, VV, Pqi, Pq;
			for (uint64_t tt = 0ULL; tt < Nt - (window - 1ULL); tt++) {

				loadSystemMatrix(storeData, sparseS, tt, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
					sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
				if (complexType == 3)
					xtr = join(0, real(xt(span, tt, NN)), imag(xt(span, tt, NN)));
				else
					xtr = real(xt(span, tt, NN));
				if (complexType == 1 || complexType == 2)
					xti = imag(xt(span, tt, NN));
				computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
				//if (complexType <= 2)
				//	computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
				//else
				//	computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
				SS = getMeas(regularization, window, m0, nMeas, NN, Nm, tt, complexType, hnU, Ly);

				Pq = matrixInversionP(sparseQ, useF, complexF, complexType, xtr, xti, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, false);

				if (tt <= 2ULL) {
					a0 = xtr;
					//a0 = randn(Nm, 1);
					//if (complexType == 2)
					//	a0i = randn(Nm, 1);
				}
				//temp = matrixInversionP(sparseQ, useF, complexF, complexType, a0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, false);
				if (complexType == 3) {
					temp = matmul3(S, Si, a0, tt % hnU);
				}
				else
					temp = matmul(S[tt % hnU], a0);
				if (sparseR)
					temp = matmul(R[tt % sizeR], temp);
				else
					temp = R[tt % sizeR] * temp;
				if (complexType == 3) {
					temp = matmul3(S, Si, temp, tt % hnU, true) + matrixInversionP(sparseQ, useF, complexF, complexType, a0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, false);
				}
				else
					temp = matmul(S[tt % hnU], temp, AF_MAT_TRANS) + matrixInversionP(sparseQ, useF, complexF, complexType, a0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, false);
				if (sparseR)
					temp2 = matmul(R[tt % sizeR], SS);
				else
					temp2 = R[tt % sizeR] * SS;
				if (complexType == 3) {
					temp2 = matmul3(S, Si, temp2, tt % hnU, true) + Pq;
				}
				else
					temp2 = matmul(S[tt % hnU], temp2, AF_MAT_TRANS) + Pq;
				r0 = temp2 - temp;
				p0 = r0;
				array V = constant(0.f, xtr.dims(0), cgIter);
				array P = constant(0.f, xtr.dims(0), cgIter);
				array D = constant(0.f, cgIter, 1);
				int64_t il = -1;
				//mexPrintf("p0.dims(0) = %d\n", p0.dims(0));
				//mexPrintf("p0.dims(1) = %d\n", p0.dims(1));
				//mexEvalString("pause(.0001);");

				for (uint32_t ii = 0; ii < cgIter; ii++) {
					il++;
					P(span, ii) = p0;
					V(span, ii) = r0;
					if (complexType == 3) {
						temp = matmul3(S, Si, p0, tt % hnU);
					}
					else
						temp = matmul(S[tt % hnU], p0);
					if (sparseR)
						temp = matmul(R[tt % sizeR], temp);
					else
						temp = R[tt % sizeR] * temp;
					//temp = matmul(S[tt % hnU], temp, AF_MAT_TRANS) + (Q[0](span, NN) * p0 - Q[0](span, NN) * matmul(Pplus, solve(identity(Pplus.dims(1), Pplus.dims(1)) + matmulTN(Pplus, tile(Q[0](span, NN), 1, Pplus.dims(1)) * Pplus), matmulTN(Pplus, Q[0](span, NN) * p0))));
					if (complexType == 3) {
						temp = matmul3(S, Si, temp, tt % hnU, true) + matrixInversionP(sparseQ, useF, complexF, complexType, p0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, false);
					}
					else
						temp = matmul(S[tt % hnU], temp, AF_MAT_TRANS) + matrixInversionP(sparseQ, useF, complexF, complexType, p0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, false);
					if (sparseR)
						temp2 = matmul(R[tt % sizeR], SS);
					else
						temp2 = R[tt % sizeR] * SS;
					const array rk = temp;
					//D(ii) = 1.f / sqrt(dot(p0, rk));
					//const array gam = dot(r0, r0) / dot(p0, rk);
					//a0 += tile(gam, p0.dims(0), 1) * p0;
					//array ra = r0 - tile(gam, rk.dims(0), 1) * rk;
					//const array be = -dot(ra, ra) / dot(r0, r0);
					//p0 = ra - tile(be, p0.dims(0), 1) * p0;
					//r0 = ra;
					D(ii) = 1.f / sqrt(dot(p0, rk));
					const array gam = dot(r0, r0) / dot(p0, rk);
					a0 = a0 + tile(gam, p0.dims(0), 1) * p0;
					array ra = r0 - tile(gam, rk.dims(0), 1) * rk;
					if (forceOrthogonalization) {
						ra = MGSOG(join(1, V(span, seq(0, ii)), ra));
						ra = ra(span, end);
					}
					const array be = -dot(ra, ra) / dot(r0, r0);
					p0 = ra - tile(be, p0.dims(0), 1) * p0;
					//if (forceOrthogonalization) {
					//	array pa = MGSOG(join(1, PP(span, seq(0, ii)), p0));
					//	p0 = pa(span, end);
					//}
					r0 = ra;
					if (norm(r0) < cgThreshold)
						break;
				}
				//if (computeConsistency && tt >= initialSteps) {
				//	storeConsistency(tt, Nt, stepSize, eps, SS, a0, cc, v, sparseR, regularization, Pplus, epsP, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, 1);
				//}
				//mexPrintf("P.dims(0) = %d\n", P.dims(0));
				//mexPrintf("P.dims(1) = %d\n", P.dims(1));
				//mexEvalString("pause(.0001);");

				Pplus = P * tile(D.T(), P.dims(0), 1);
				//mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
				//mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
				//mexPrintf("il = %d\n", il);
				//mexEvalString("pause(.0001);");
				Pplus = Pplus(span, seq(0, il));
				xtr = a0;
				if (complexType == 0)
					xt(span, tt + 1, NN) = xtr;
				else if (complexType == 1 || complexType == 2) {
					if (complexRef && augType > 0)
						SS = getMeas(regularization, window, m0, nMeas, NN, Nm, tt, complexType, hnU, Li, true);
					else
						SS = getMeas(regularization, window, m0, nMeas, NN, Nm, tt, complexType, hnU, Ly, true);

					Pq = matrixInversionP(sparseQ, useF, complexF, complexType, xtr, xti, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, true, true);

					if (tt <= 2ULL) {
						a0i = xti;
					}
					if (complexS)
						temp = matmul(Si[tt % hnU], a0i);
					else
						temp = matmul(S[tt % hnU], a0i);
					if (complexType == 2) {
						if (sparseR)
							temp = matmul(Ri[tt % sizeR], temp);
						else
							temp = Ri[tt % sizeR] * temp;
					}
					else {
						if (sparseR)
							temp = matmul(R[tt % sizeR], temp);
						else
							temp = R[tt % sizeR] * temp;
					}
					if (complexS)
						temp = matmul(Si[tt % hnU], temp, AF_MAT_TRANS) + matrixInversionP(sparseQ, useF, complexF, complexType, a0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, true, true);
					else
						temp = matmul(S[tt % hnU], temp, AF_MAT_TRANS) + matrixInversionP(sparseQ, useF, complexF, complexType, a0, a0i, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, true, true);
					if (complexType == 2) {
						if (sparseR)
							temp2 = matmul(Ri[tt % sizeR], SS);
						else
							temp2 = Ri[tt % sizeR] * SS;
					}
					else {
						if (sparseR)
							temp2 = matmul(R[tt % sizeR], SS);
						else
							temp2 = R[tt % sizeR] * SS;
					}
					if (complexS)
						temp2 = matmul(Si[tt % hnU], temp2, AF_MAT_TRANS) + Pq;
					else
						temp2 = matmul(S[tt % hnU], temp2, AF_MAT_TRANS) + Pq;
					r0 = temp2 - temp;
					p0 = r0;
					array V = constant(0.f, xt.dims(0), cgIter);
					array P = constant(0.f, xt.dims(0), cgIter);
					array D = constant(0.f, cgIter, 1);
					int64_t il = -1;
					//mexPrintf("p0.dims(0) = %d\n", p0.dims(0));
					//mexPrintf("p0.dims(1) = %d\n", p0.dims(1));
					//mexEvalString("pause(.0001);");

					for (uint32_t ii = 0; ii < cgIter; ii++) {
						il++;
						P(span, ii) = p0;
						V(span, ii) = r0;
						if (complexS)
							temp = matmul(Si[tt % hnU], p0);
						else
							temp = matmul(S[tt % hnU], p0);
						if (complexType == 2) {
							if (sparseR)
								temp = matmul(Ri[tt % sizeR], temp);
							else
								temp = Ri[tt % sizeR] * temp;
						}
						else {
							if (sparseR)
								temp = matmul(R[tt % sizeR], temp);
							else
								temp = R[tt % sizeR] * temp;
						}
						if (complexS)
							temp = matmul(Si[tt % hnU], temp, AF_MAT_TRANS) + matrixInversionP(sparseQ, useF, complexF, complexType, p0, p0, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, true, true);
						else
							temp = matmul(S[tt % hnU], temp, AF_MAT_TRANS) + matrixInversionP(sparseQ, useF, complexF, complexType, p0, p0, Q, Qi, F, Fi, NN, tt % sizeQ, tt % sizeF, Pplus, Pplusi, true, true);
						if (complexType == 2) {
							if (sparseR)
								temp2 = matmul(Ri[tt % sizeR], SS);
							else
								temp2 = Ri[tt % sizeR] * SS;
						}
						else {
							if (sparseR)
								temp2 = matmul(R[tt % sizeR], SS);
							else
								temp2 = R[tt % sizeR] * SS;
						}
						const array rk = temp;
						D(ii) = 1.f / sqrt(dot(p0, rk));
						const array gam = dot(r0, r0) / dot(p0, rk);
						a0i = a0i + tile(gam, p0.dims(0), 1) * p0;
						array ra = r0 - tile(gam, rk.dims(0), 1) * rk;
						if (forceOrthogonalization) {
							ra = MGSOG(join(1, V(span, seq(0, ii)), ra));
							ra = ra(span, end);
						}
						const array be = -dot(ra, ra) / dot(r0, r0);
						p0 = ra - tile(be, p0.dims(0), 1) * p0;
						//if (forceOrthogonalization) {
						//	array pa = MGSOG(join(1, PP(span, seq(0, ii)), p0));
						//	p0 = pa(span, end);
						//}
						r0 = ra;
						if (norm(r0) < cgThreshold)
							break;
					}
					//if (computeConsistency && tt >= initialSteps) {
					//	storeConsistency(tt, Nt, stepSize, epsi, SS, a0i, cc, v, sparseR, regularization, Pplusi, epsPi, computeBayesianP, R[tt % sizeR], RR, S[tt % hnU], complexType, Si, hnU, 1);
					//}

					Pplusi = P * tile(D.T(), P.dims(0), 1);
					//mexPrintf("Pplusi.dims(0) = %d\n", Pplusi.dims(0));
					//mexPrintf("Pplusi.dims(1) = %d\n", Pplusi.dims(1));
					//mexPrintf("Iil = %d\n", il);
					//mexEvalString("pause(.0001);");
					Pplusi = Pplusi(span, seq(0, il));
					xti = a0i;
					xt(span, tt + 1, NN) = complex(xtr, xti);
				}
				else if (complexType == 3) {
					if (kineticModel)
						xt(span, tt + 1, NN) = complex(join(0, xtr(seq(0, imDim - 1)), xtr(seq(imDimU, imDimU + imDim - 1))), join(0, xtr(seq(imDim, imDimU - 1)), xtr(seq(imDimU + imDim, end))));
					else
						xt(span, tt + 1, NN) = complex(xtr(seq(0, xtr.dims(0) / 2 - 1)), xtr(seq(xtr.dims(0) / 2, end)));
				}
				if (useSmoother && ((tt == N_lag - 1) || (tt >= N_lag - 1 && jg == skip) || tt == (Nt - 1))) {
					jg = 0;
					int64_t ww = 0;
					int64_t ll = tt % N_lag;
					int64_t jj = tt % N_lag;
					int64_t ji = tt % N_lag;
					if (complexType != 2)
						ji = 0;
					int64_t qq = tt % sizeQ;
					//int64_t q2 = qq;
					int64_t ff = tt % sizeF;
					if (complexType == 3)
						xlt1(span, end) = join(0, real(xt(seq(0, imDim - 1), tt + 1, NN)), imag(xt(seq(0, imDim - 1), tt + 1, NN)));
					else
						xlt1(span, end) = xt(seq(0, imDim - 1), tt + 1, NN);
					array xTemp, xTempi, Pqi, Pq;
					if (DEBUG) {
						mexPrintf("N_lag = %d\n", N_lag);
						mexPrintf("tt = %d\n", tt);
						mexPrintf("xlt1.dims(0) = %d\n", xlt1.dims(0));
						mexPrintf("xlt1.dims(1) = %d\n", xlt1.dims(1));
						mexEvalString("pause(.0001);");
					}
					for (int64_t to = N_lag - 1; to >= 0; to--) {
						if (complexType == 3)
							xtr = join(0, real(xt(span, tt - ww, NN)), imag(xt(span, tt - ww, NN)));
						else
							xtr = real(xt(span, tt - ww, NN));
						if (complexType == 1 || complexType == 2)
							xti = imag(xt(span, tt - ww, NN));
						if (useF || useG || useU)
							computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
						xTemp = real(xlt1(span, to + 1)) - xtr;
						if (complexType == 2 || complexType == 1)
							xTempi = imag(xlt1(span, to + 1)) - xti;
						//Pq = xTemp;
						//if (complexType == 2 || complexType == 1)
						//	Pqi = xTempi;
						//mexPrintf("xTemp.dims(0) = %d\n", xTemp.dims(0));
						//mexPrintf("xTemp.dims(1) = %d\n", xTemp.dims(1));
						//mexPrintf("xTemp.summa = %f\n", af::sum<float>(flat(xTemp)));
						//mexEvalString("pause(.0001);");
						Pq = matrixInversionP(sparseQ, useF, complexF, complexType, xTemp, xTempi, Q, Qi, F, Fi, NN, qq, ff, KS[jj], KSi[ji]);
						if (complexType == 2 || complexType == 1) {
							Pqi = Pq(span, end);
							Pq = Pq(span, 0);
						}
						//mexPrintf("Pq3.dims(0) = %d\n", Pq.dims(0));
						//mexPrintf("Pq3.dims(1) = %d\n", Pq.dims(1));
						//mexPrintf("Pq.summa = %f\n", af::sum<float>(flat(Pq)));
						//mexEvalString("pause(.0001);");
						if (useF) {
							Pq = matmul(F[ff], Pq, AF_MAT_TRANS);
							if (complexType == 2)
								if (complexF)
									Pqi = matmul(Fi[ff], Pqi, AF_MAT_TRANS);
								else
									Pqi = matmul(F[ff], Pqi, AF_MAT_TRANS);
							else if (complexType == 1)
								Pqi = matmul(F[ff], Pqi, AF_MAT_TRANS);
						}
						Pq = matmul(KS[jj], KS[jj].T(), Pq);
						if (complexType == 2)
							Pqi = matmul(KSi[ji], KSi[ji].T(), Pqi);
						else if (complexType == 1)
							Pqi = matmul(KS[jj], KS[jj].T(), Pqi);
						if (complexType == 0)
							xlt1(span, to) = xt(seq(0, imDim - 1), tt - ww) + Pq;
						else if (complexType == 1)
							xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), tt - ww, NN)) + Pq, imag(xt(seq(0, imDim - 1), tt - ww, NN)) + Pqi);
						else if (complexType == 2)
							xlt1(span, to) = complex(real(xt(seq(0, imDim - 1), tt - ww, NN)) + Pq, imag(xt(seq(0, imDim - 1), tt - ww, NN)) + Pqi);
						else if (complexType == 3)
							xlt1(span, to) = join(0, real(xt(seq(0, imDim - 1), tt - ww, NN)), imag(xt(seq(0, imDim - 1), tt - ww, NN))) + Pq;
						//mexPrintf("xlt1(span, to + 1).summa = %f\n", af::sum<float>(flat(real(xlt1(span, to + 1)))));
						//mexPrintf("xlt1(span, to + 1).summa = %f\n", af::sum<float>(flat(real(xt(seq(0, imDim - 1), tt - ww + 1, NN)))));
						//mexPrintf("ww = %d\n", ww);
						ww++;
						jj--;
						ji--;
						ll++;
						qq--;
						ff--;
						if (jj < 0)
							jj = N_lag - 1;
						if (ji < 0)
							if (complexType == 2)
								ji = N_lag - 1;
							else
								ji = 0;
						if (ll == N_lag)
							ll = 0;
						if (ff < 0)
							ff = 0LL;
						if (qq < 0)
							qq = 0LL;
					}
					if (complexType == 3)
						xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = complex(xlt1(seq(0, xlt1.dims(0) / 2 - 1), seq(0, end - 1LL)), xlt1(seq(xlt1.dims(0) / 2, end), seq(0, end - 1LL)));
					else
						xlt(span, seq(tt + 1LL - (N_lag), tt), NN) = xlt1(span, seq(0, N_lag - 1LL));
				}
				if (regularization > 2) {
					computeDenoising(xt, imDim, tt, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
						huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, Type, Pred);
					if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
						mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
						break;
					}
				}
			}
			if (storeCovariance == 2)
				if (complexType == 2)
					if (kineticModel)
						P1(span, seq(0, Pplus.dims(1) - 1)) = complex(Pplus(seq(0, imDim - 1), span), Pplusi(seq(0, imDim - 1), span));
					else
						P1(span, seq(0, Pplus.dims(1) - 1)) = complex(Pplus, Pplusi);
				else if (complexType == 3)
					P1(span, seq(0, Pplus.dims(1) - 1)) = complex(Pplus(seq(0, imDim - 1), span), Pplus(seq(imDim, end), span));
				else
					if (kineticModel)
						P1(span, seq(0, Pplus.dims(1) - 1)) = Pplus(seq(0, imDimU - 1), span);
					else
						P1(span, seq(0, Pplus.dims(1) - 1)) = Pplus;
			//if (computeConsistency) {
			//	computeConsistencyTests(cc, Nt, stepSize, initialSteps, v, Nm, eps, epsP, computeBayesianP, complexType, vvi, epsi, epsPi);
			//}
		}
		else if (algorithm == 12) {
			Pminus = Pplus;
			if (complexType == 2)
				Pminusi = Pplusi;
			if (DEBUG) {
				mexPrintf("sizeQ = %d\n", sizeQ);
				mexEvalString("pause(.0001);");
			}
			for (uint64_t tt = 0ULL; tt < Nt - (window - 1ULL); tt++) {

				//jg++;
				loadSystemMatrix(storeData, sparseS, tt, window, hn, hnU, Nm, complexS, lSize, complexType, regularization, imDimN,
					sCol, S1, S2, Svalues, Svaluesi, Srow, Scol, sCols, sRows, Lvalues, Lcol, LL, SS3, SS4, S, Si);
				if (complexType == 3)
					xtr = join(0, real(xt(span, tt, NN)), imag(xt(span, tt, NN)));
				else
					xtr = real(xt(span, tt, NN));
				if (complexType == 1 || complexType == 2)
					xti = imag(xt(span, tt, NN));
				computeAPrioriX(useF, useU, useG, useKineticModel, algorithm, xtr, xti, imDim, F, Fi, complexType, G, Gi, u, complexF, complexG, tt, sizeF, sizeG, sizeU);
				array T;
				if (useF)
					T = matmul(F[tt % sizeF], Pplus);
				else
					T = Pplus;
				//T += tile(Q[tt % sizeQ], 1, S[tt % hnU].dims(0)) * transpose(S[tt % hnU]);
				T += Q[tt % sizeQ];
				//array testi = (matmul(S[tt % hnU], T) + tile(R[tt % sizeR], 1, T.dims(1)));
				//af::eval(testi);
				//mexPrintf("testi.summa = %f\n", af::sum<float>(flat(testi)));
				//mexPrintf("testi.min = %f\n", af::min<float>(flat(testi)));
				//mexEvalString("pause(.0001);");
				if (sparseR)
					if (complexType == 3)
						KG = transpose(solve(transpose(matmul3(S, Si, T, tt % hnU) + R[tt % sizeR]), T.T()));
					else
						KG = transpose(solve(transpose(matmul(S[tt % hnU], T) + R[tt % sizeR]), T.T()));
				else
					if (complexType == 3)
						KG = transpose(solve(transpose(matmul3(S, Si, T, tt % hnU) + diag(R[tt % sizeR], 0, false)), T.T()));
					else
						KG = transpose(solve(transpose(matmul(S[tt % hnU], T) + diag(R[tt % sizeR], 0, false)), T.T()));
					//KG = matmul(T, testi);
				if (complexType <= 2)
					computeInnovation(regularization, SS, window, real(m0), nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
				else
					computeInnovation(regularization, SS, window, m0, nMeas, NN, Nm, tt, S[tt % hnU], xtr, Ly, Si, complexType, hnU);
				xtr += matmul(KG, SS);
				//Pplus = 2.f * Pplus - Pminus + tile(Q[tt % sizeQ], 1, S[tt % hnU].dims(0)) * transpose(S[tt % hnU]);
				Pplus = 2.f * Pplus - Pminus + Q[tt % sizeQ];
				if (complexType == 3) {
					Pplus = Pplus - matmul(KG, matmul3(S, Si, Pplus, tt % hnU));
					Pminus = T - matmul(KG, matmul3(S, Si, T, tt % hnU));
				}
				else {
					Pplus = Pplus - matmul(KG, S[tt % hnU], Pplus);
					Pminus = T - matmul(KG, S[tt % hnU], T);
				}

				if (complexType == 0)
					xt(span, tt + 1, NN) = xtr;
				else if (complexType == 1 || complexType == 2) {
					if (complexS)
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, Si[tt % hnU], xti, Ly, Si, complexType, hnU);
					else
						if (complexRef && augType > 0 && regularization == 1)
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Li, Si, complexType, hnU);
						else
							computeInnovation(regularization, SS, window, imag(m0), nMeas, NN, Nm, tt, S[tt % hnU], xti, Ly, Si, complexType, hnU);
					if (complexType == 2) {
						if (useF)
							T = matmul(F[tt % sizeF], Pplusi);
						else
							T = Pplusi;
						T += Qi[tt % sizeQ];
						if (complexS)
							if (sparseR)
								KG = transpose(solve(transpose(matmul(Si[tt % hnU], T) + Ri[tt % sizeR]), T.T()));
							else
								KG = transpose(solve(transpose(matmul(Si[tt % hnU], T) + diag(Ri[tt % sizeR], 0, false)), T.T()));
						else
						if (sparseR)
							KG = transpose(solve(transpose(matmul(S[tt % hnU], T) + Ri[tt % sizeR]), T.T()));
						else
							KG = transpose(solve(transpose(matmul(S[tt % hnU], T) + diag(Ri[tt % sizeR], 0, false)), T.T()));
						Pplusi = 2.f * Pplusi - Pminusi + Qi[tt % sizeQ];
						if (complexS) {
							//Pplusi = 2.f * Pplusi - Pminusi + tile(Qi[tt % sizeQ], 1, Si[tt % hnU].dims(0)) * transpose(Si[tt % hnU]);
							Pplusi = Pplusi - matmul(KG, Si[tt % hnU], Pplusi);
							Pminusi = T - matmul(KG, Si[tt % hnU], T);
						}
						else {
							//Pplusi = 2.f * Pplusi - Pminusi + tile(Qi[tt % sizeQ], 1, S[tt % hnU].dims(0)) * transpose(S[tt % hnU]);
							Pplusi = Pplus - matmul(KG, S[tt % hnU], Pplusi);
							Pminusi = T - matmul(KG, S[tt % hnU], T);
						}
					}
					xti += matmul(KG, SS);
					xt(span, tt + 1, NN) = complex(xtr, xti);
				}
				else if (complexType == 3)
					if (kineticModel)
						xt(span, oo + 1, NN) = complex(join(0, xtr(seq(0, imDim - 1)), xtr(seq(imDimU, imDimU + imDim - 1))), join(0, xtr(seq(imDim, imDimU - 1)), xtr(seq(imDimU + imDim, end))));
					else
						xt(span, oo + 1, NN) = complex(xtr(seq(0, xtr.dims(0) / 2 - 1)), xtr(seq(xtr.dims(0) / 2, end)));
				if (regularization > 2) {
					computeDenoising(xt, imDim, tt, NN, complexType, regularization, Pplus, Pplusi, prior, nIter, Nx, Ny, DimZ, TV, TVi, Ndx, Ndy, Ndz, gamma, beta, betac,
						huberDelta, weightsHuber, weightsQuad, LL, complexRef, Li, TGV, 1);
					if (sum<float>(af::isNaN(real(xt(seq(0, imDim - 1), oo + 1, NN)))) > 0) {
						mexPrintf("NaN values detected in the regularized estimates, aborting.\n");
						break;
					}
				}
			}
		}
	}
	//if (useKineticModel && NDimZ > 1) {
	//	array apuX = constant(0.f, imDim, xt.dims(1), NDimZ);
	//	for (uint64_t NN = 0; NN < NDimZ; NN++)
	//		apuX(span, span, NN) = xt(seq(0, imDim - 1), span, NN);
	//	xt = apuX;
	//}
	//if (useSmoother)
	//	xlt(span, end) = xt(span, end);
}


void kfilter(const mxArray* options, float* out, float* out2, float* outS, float* out2S, float* outP, float* outP2) {

	/* Load various parameters */
	// Are the measurements complex
	const bool complexMeas = (bool)mxGetScalar(mxGetField(options, 0, "complexMeasurements"));
	if (DEBUG) {
		mexPrintf("complexMeas = %u\n", complexMeas);
		mexEvalString("pause(.0001);");
	}
	// Type of algorithm used (regular KF, information filter, etc.)
	const uint32_t algorithm = (uint32_t)mxGetScalar(mxGetField(options, 0, "algorithm"));
	// What kind of complex data is used (no complex data, only separate estimates, seperate gains, totally complex)
	const uint8_t complexType = (uint8_t)mxGetScalar(mxGetField(options, 0, "complexType"));
	// Is fading memory used
	const bool fadingMemory = (bool)mxGetScalar(mxGetField(options, 0, "fadingMemory"));
	// Is KS used
	const bool useSmoother = (bool)mxGetScalar(mxGetField(options, 0, "useKS"));
	// Is the approximated KS used (take only the diagonal from the a priori covariance)
	const bool approximateKS = (bool)mxGetScalar(mxGetField(options, 0, "approximateKS"));
	// Is the steady state KS used
	const bool steadyS = (bool)mxGetScalar(mxGetField(options, 0, "steadyKS"));
	// Is the steady state KF used
	const bool steadyKF = (bool)mxGetScalar(mxGetField(options, 0, "steadyKF"));
	// Is full 3D or pseudo-3D used if 3D data?
	const bool use3D = (bool)mxGetScalar(mxGetField(options, 0, "use3D"));
	// Perform consistency tests?
	const bool computeConsistency = (bool)mxGetScalar(mxGetField(options, 0, "computeConsistency"));
	// Perform consistency tests?
	const bool computeBayesianP = (bool)mxGetScalar(mxGetField(options, 0, "computeBayesianP"));
	// compute the initial value for KF
	const bool computeInitialValue = (bool)mxGetScalar(mxGetField(options, 0, "computeInitialValue"));
	// Force orthogonalization in CGKF and CGVKF
	const bool forceOrthogonalization = (bool)mxGetScalar(mxGetField(options, 0, "forceOrthogonalization"));
	// Use either the ensemble mean as the next a priori estimate or the previous full ensemble a posteriori estimate
	const bool useEnsembleMean = (bool)mxGetScalar(mxGetField(options, 0, "useEnsembleMean"));
	// Number of time steps
	const uint64_t Nt = (uint64_t)mxGetScalar(mxGetField(options, 0, "Nt"));
	// Total number of pixels/voxels
	const uint64_t imDim = (uint64_t)mxGetScalar(mxGetField(options, 0, "N"));
	// Step size of the time average autocorrelation test
	const uint64_t stepSize = (uint64_t)mxGetScalar(mxGetField(options, 0, "stepSize"));
	// Number of pixels in the X/Y/Z-dimension
	const uint32_t Nx = (uint32_t)mxGetScalar(mxGetField(options, 0, "Nx"));
	const uint32_t Ny = (uint32_t)mxGetScalar(mxGetField(options, 0, "Ny"));
	const uint32_t Nz = (uint32_t)mxGetScalar(mxGetField(options, 0, "Nz"));
	// Number of time steps potentially skipped in KS
	const uint32_t skip = (uint32_t)mxGetScalar(mxGetField(options, 0, "sSkip"));
	// Type of regularization used
	const uint32_t regularization = (uint32_t)mxGetScalar(mxGetField(options, 0, "regularization"));
	// Number of measurements per time step
	const uint64_t Nm = (uint64_t)mxGetScalar(mxGetField(options, 0, "Nm"));
	// Number of time steps included (sliding window length)
	const uint64_t window = (uint64_t)mxGetScalar(mxGetField(options, 0, "window"));
	// Ensemble size
	const uint64_t ensembleSize = (uint64_t)mxGetScalar(mxGetField(options, 0, "ensembleSize"));
	// Is the kinetic model used
	bool useKineticModel = (bool)mxGetScalar(mxGetField(options, 0, "useKinematicModel"));
	// Number of pre-iterations for the covariance
	const uint64_t covIter = (uint64_t)mxGetScalar(mxGetField(options, 0, "covIter"));
	// Number of pre-iterations for the consistency tests
	const uint64_t initialSteps = (uint64_t)mxGetScalar(mxGetField(options, 0, "consistencyIter"));
	// Smoother lag
	const uint64_t N_lag = (uint64_t)mxGetScalar(mxGetField(options, 0, "nLag"));
	// Neighborhood size
	const uint32_t Ndx = (uint32_t)mxGetScalar(mxGetField(options, 0, "Ndx"));
	const uint32_t Ndy = (uint32_t)mxGetScalar(mxGetField(options, 0, "Ndy"));
	const uint32_t Ndz = (uint32_t)mxGetScalar(mxGetField(options, 0, "Ndz"));
	// Type of prior used with regularization type 3
	const uint32_t prior = (uint32_t)mxGetScalar(mxGetField(options, 0, "prior"));
	// Number of denoising iterations (regularization types 3 and 4)
	const uint32_t nIter = (uint32_t)mxGetScalar(mxGetField(options, 0, "nIterDenoise"));
	// Number of CG iterations (CGKF and CGVKF)
	const uint32_t cgIter = (uint32_t)mxGetScalar(mxGetField(options, 0, "cgIter"));
	// Type of augmentation (no anatomical reference image or anatomical reference image)
	const uint32_t augType = (uint32_t)mxGetScalar(mxGetField(options, 0, "augType"));
	// Whether the covariance matrix is stored
	const uint8_t storeCovariance = (uint8_t)mxGetScalar(mxGetField(options, 0, "storeCovariance"));
	// CG iteration stopping threshold
	const double cgThreshold = mxGetScalar(mxGetField(options, 0, "cgThreshold"));
	// Regularization parameter (real part)
	const float beta = (float)mxGetScalar(mxGetField(options, 0, "beta"));
	// Regularization parameter (imaginary part)
	const float betac = (float)mxGetScalar(mxGetField(options, 0, "betaI"));
	// Step size for regularization type 3
	const float gamma = (float)mxGetScalar(mxGetField(options, 0, "gamma"));
	// Delta value for Huber prior
	const float huberDelta = (float)mxGetScalar(mxGetField(options, 0, "huberDelta"));
	// Alpha value for the fading memory filter
	const float fadingAlpha = (float)mxGetScalar(mxGetField(options, 0, "fadingAlpha"));
	// Noise variance for the augmented portion with regularization 1
	const float RR = (float)mxGetScalar(mxGetField(options, 0, "RR"));
	// Include covariance matrix as a weighting matrix in denoising
	const bool regType = (bool)mxGetScalar(mxGetField(options, 0, "includeCovariance"));
	TVdata TV, TVi;
	// Whether anatomical reference image is used with regularization types 3 or 4
	TV.TV_use_anatomical = (bool)mxGetScalar(mxGetField(options, 0, "useAnatomical"));
	// Whether separate anatomical reference images are used for real and imaginary parts (true) or real part for both (false)
	const bool complexRef = (bool)mxGetScalar(mxGetField(options, 0, "complexRef"));
	if (DEBUG) {
		mexPrintf("complexRef = %u\n", complexRef);
		mexEvalString("pause(.0001);");
	}
	uint64_t DimZ = 1ULL;
	if (!use3D && Nz > 1)
		DimZ = Nz;
	uint64_t imDimN = imDim;
	if (useKineticModel)
		imDimN *= 2ULL;
	uint64_t imDimU = imDimN;
	if (complexType == 3)
		imDimU *= 2ULL;
	array weightsQuad, weightsHuber;
	if ((prior > 0 && prior < 5 && regularization == 3) || (regularization == 4 && TV.TV_use_anatomical)) {
		TV.SATVPhi = (float)mxGetScalar(mxGetField(options, 0, "SATVPhi"));
		TV.TVsmoothing = (float)mxGetScalar(mxGetField(options, 0, "TVsmoothing"));
		TV.APLSsmoothing = TV.TVsmoothing;
		TV.T = (float)mxGetScalar(mxGetField(options, 0, "C"));
		TV.eta = (float)mxGetScalar(mxGetField(options, 0, "eta"));
		TV.tau = (float)mxGetScalar(mxGetField(options, 0, "tau"));
		if ((regularization == 3 && TV.TV_use_anatomical) || (regularization == 4 && TV.TV_use_anatomical)) {
			// Anatomical reference image (real part)
			TV.reference_image = array(imDim, (float*)mxGetData(mxGetField(options, 0, "referenceImage")));
			if (regularization == 3) {
				const size_t sSize = mxGetNumberOfElements(mxGetField(options, 0, "s1"));
				TV.s1 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s1")), afHost);
				TV.s2 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s2")), afHost);
				TV.s3 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s3")), afHost);
				TV.s4 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s4")), afHost);
				TV.s5 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s5")), afHost);
				TV.s6 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s6")), afHost);
				TV.s7 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s7")), afHost);
				TV.s8 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s8")), afHost);
				TV.s9 = af::array(sSize, (float*)mxGetData(mxGetField(options, 0, "s9")), afHost);
			}
		}
		if (complexRef) {
			TVi.TV_use_anatomical = TV.TV_use_anatomical;
			TVi.SATVPhi = (float)mxGetScalar(mxGetField(options, 0, "SATVPhiI"));
			TVi.TVsmoothing = (float)mxGetScalar(mxGetField(options, 0, "TVsmoothingI"));
			TVi.APLSsmoothing = TVi.TVsmoothing;
			TVi.T = (float)mxGetScalar(mxGetField(options, 0, "CI"));
			TVi.eta = (float)mxGetScalar(mxGetField(options, 0, "etaI"));
			TVi.tau = (float)mxGetScalar(mxGetField(options, 0, "tau"));
			if ((regularization == 3 && TVi.TV_use_anatomical) || (regularization == 4 && TV.TV_use_anatomical)) {
				// Anatomical reference image (imaginary part)
				TVi.reference_image = array(imDim, (float*)mxGetImagData(mxGetField(options, imDim, "referenceImage")));
				if (regularization == 3) {
					const size_t sSize = mxGetNumberOfElements(mxGetField(options, 0, "s1"));
					TVi.s1 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s1")), afHost);
					TVi.s2 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s2")), afHost);
					TVi.s3 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s3")), afHost);
					TVi.s4 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s4")), afHost);
					TVi.s5 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s5")), afHost);
					TVi.s6 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s6")), afHost);
					TVi.s7 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s7")), afHost);
					TVi.s8 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s8")), afHost);
					TVi.s9 = af::array(sSize, (float*)mxGetData(mxGetField(options, imDim, "s9")), afHost);
				}
			}
		}
		else
			TVi = TV;
	}
	else if ((prior == 5 && regularization == 3)) {
		const size_t wN = mxGetNumberOfElements(mxGetField(options, 0, "weights"));
		weightsQuad = af::array(wN, (float*)mxGetData(mxGetField(options, 0, "weights")), afHost);
	}
	else if ((prior == 6 && regularization == 3)) {
		const size_t wN = mxGetNumberOfElements(mxGetField(options, 0, "weights"));
		weightsHuber = af::array(wN, (float*)mxGetData(mxGetField(options, 0, "weights")), afHost);
	}
	TGVdata TGV;
	TGV.lambda1 = (float)mxGetScalar(mxGetField(options, 0, "lambda1"));
	TGV.lambda2 = (float)mxGetScalar(mxGetField(options, 0, "lambda2"));
	TGV.prox = (float)mxGetScalar(mxGetField(options, 0, "proximalValue"));
	TGV.rho = (float)mxGetScalar(mxGetField(options, 0, "relaxationParameter"));
	array m0;
	array xt;
	if (DEBUG) {
		mexPrintf("imDim = %u\n", imDim);
		mexPrintf("imDimN = %u\n", imDimN);
		mexEvalString("pause(.0001);");
	}
	if (complexMeas) {
		xt = constant(0.f, imDimN, Nt + 1ULL - (window - 1ULL), DimZ, c32);
		//const float* HH1 = loadFloats(options, "m0");
		//const float* HH1 = (float*)mxGetComplexSingles(mxGetField(options, 0, "m0"));
		// Measurements (real part)
		const float* HH1 = (float*)mxGetData(mxGetField(options, 0, "m0"));
		// Measurements (imaginary part)
		const float* HH2 = (float*)mxGetImagData(mxGetField(options, 0, "m0"));
		// Initial estimate (real part)
		const float* x0r = (float*)mxGetData(mxGetField(options, 0, "x0"));
		// Initial estimate (imaginary part)
		const float* x0i = (float*)mxGetImagData(mxGetField(options, 0, "x0"));
		const size_t numEle = mxGetNumberOfElements(mxGetField(options, 0, "m0"));
		const size_t numRowS = mxGetM(mxGetField(options, 0, "m0"));
		const size_t numColS = mxGetN(mxGetField(options, 0, "m0"));
		const size_t numRowsX = mxGetM(mxGetField(options, 0, "x0"));
		const size_t numColsX = mxGetN(mxGetField(options, 0, "x0"));
		//mexPrintf("numColS = %u\n", numColS);
		//mexPrintf("numRowS = %u\n", numRowS);
		if (DEBUG) {
			mexPrintf("numRowsX = %u\n", numRowsX);
			mexPrintf("numColsX = %u\n", numColsX);
			mexEvalString("pause(.0001);");
		}
		const size_t numSlicesX = mxGetNumberOfElements(mxGetField(options, 0, "x0")) / (numRowsX * numColsX);
		//if (complexType == 3) {
		//	m0 = join(0, array(numRowS, numColS, numEle / (numRowS * numColS), HH1), array(numRowS, numColS, numEle / (numRowS * numColS), HH2));
		//	m0 = flat(m0);
		//}
		//else
		m0 = kompleksiset(numEle, 1, HH1, HH2);
		if (numRowsX * numColsX * numSlicesX == imDimN || (numColsX == Nx && numRowsX == Ny && numSlicesX == Nz)) {
			xt(span, 0, seq(0, numSlicesX - 1)) = kompleksiset(numRowsX, numColsX, x0r, x0i, numSlicesX);
		}
		if (!use3D && Nz > 1 && numSlicesX == 1) {
			for (uint32_t kk = 1; kk < Nz; kk++)
				xt(span, 0, kk) = xt(span, 0, 0);
		}
	}
	else {
		xt = constant(0.f, imDimN, Nt + 1, DimZ, f32);
		// Measurements
		const float* HH1 = loadFloats(options, "m0");
		// Initial estimate
		const float* x0r = loadFloats(options, "x0");
		//const size_t numRowS = mxGetM(mxGetField(options, 0, "m0"));
		//const size_t numColS = mxGetN(mxGetField(options, 0, "m0"));
		const size_t numEle = mxGetNumberOfElements(mxGetField(options, 0, "m0"));
		const size_t numRowsX = mxGetM(mxGetField(options, 0, "x0"));
		const size_t numColsX = mxGetN(mxGetField(options, 0, "x0"));
		const size_t numSlicesX = mxGetNumberOfElements(mxGetField(options, 0, "x0")) / (numRowsX * numColsX);
		m0 = array(numEle, 1, HH1, afHost);
		if (numRowsX * numColsX * numSlicesX == imDimN || (numColsX == Nx && numRowsX == Ny && numSlicesX == Nz)) {
			xt(span, 0, seq(0, numSlicesX - 1)) = array(numRowsX, numColsX, numSlicesX, x0r, afHost);
		}
		if (!use3D && Nz > 1 && numSlicesX == 1) {
			for (uint32_t kk = 1; kk < Nz; kk++)
				xt(span, 0, kk) = xt(span, 0, 0);
		}
	}

	// Is the system matrix complex
	const bool complexS = (bool)mxGetScalar(mxGetField(options, 0, "complexS"));
	// Is the system matrix sparse
	const bool sparseS = (bool)mxGetScalar(mxGetField(options, 0, "sparseS"));
	// Is the regularization matrix sparse
	const bool sparseL = (bool)mxGetScalar(mxGetField(options, 0, "sparseL"));
	// Is the regularization matrix complex
	const bool complexL = (bool)mxGetScalar(mxGetField(options, 0, "complexL"));
	// If true, the system matrix is completely stored to the GPU, if false only a single time step is stored
	const bool storeData = (bool)mxGetScalar(mxGetField(options, 0, "storeData"));
	// Number of unique system matrix cycles
	const uint64_t hn = (uint64_t)mxGetScalar(mxGetField(options, 0, "matCycles"));
	uint64_t hnU = hn;
	uint64_t NmU = Nm;
	if (complexType == 3)
		NmU *= 2ULL;
	std::vector<array> S(1);
	std::vector<array> Si;
	array Svalues, Srow, Scol, Svaluesi, LL, LLi, Lvalues, Lrow, Lcol, Ly, Lyi;
	float* SS3 = nullptr, * SS4 = nullptr, * L = nullptr;
	int32_t* rowL = nullptr;
	std::vector<float> S1, S2, L1, L2;
	std::vector<int32_t> sCols, sRows, lRows, lCols;
	size_t* sCol = nullptr, * lCol = nullptr;
	size_t lSize = 0ULL;
	std::vector<array> Qvalues, Qrow, Qcol, Qvaluesi, Qvalues2, Qrow2, Qcol2, Qvaluesi2, Rvalues, Rvaluesi, Rrow, Rcol, Rvalues2, Rvaluesi2, Rrow2, Rcol2;
	if (regularization == 1 || regularization == 4 || (regularization == 3 && prior == 7)) {
		if (sparseL) {
			const size_t nNZ = mxGetNzmax(mxGetField(options, 0, "L"));
			const size_t nCol = mxGetN(mxGetField(options, 0, "L"));
			const size_t nRow = mxGetM(mxGetField(options, 0, "L"));
			if (DEBUG) {
				mexPrintf("nNZ = %u\n", nNZ);
				mexPrintf("nCol = %u\n", nCol);
				mexPrintf("nRow = %u\n", nRow);
				mexEvalString("pause(.0001);");
			}
			const double* Lt = (double*)mxGetData(mxGetField(options, 0, "L"));
			if (complexRef && complexL) {
				const double* Lti = (double*)mxGetImagData(mxGetField(options, 0, "L"));
				L2.resize(nNZ, 0.f);
				std::transform(Lti, Lti + nNZ, std::begin(L2), [&](const double& value) { return static_cast<float>(value); });
			}
			L1.resize(nNZ, 0.f);
			std::transform(Lt, Lt + nNZ, std::begin(L1), [&](const double& value) { return static_cast<float>(value); });
			const size_t* lRow = reinterpret_cast<size_t*>(mxGetIr(mxGetField(options, 0, "L")));
			lRows.resize(nNZ, 0);
			std::transform(lRow, lRow + nNZ, std::begin(lRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
			lCol = reinterpret_cast<size_t*>(mxGetJc(mxGetField(options, 0, "L")));
			lSize = imDim;
			Lcol = array(nNZ, 1, lRows.data(), afHost);
			if (complexRef && complexL)
				Lvalues = kompleksiset(nNZ, 1, L1.data(), L2.data());
			else
				Lvalues = array(nNZ, L1.data(), afHost);
			if (DEBUG) {
				mexPrintf("Lvalues.dims(0) = %u\n", Lvalues.dims(0));
				mexEvalString("pause(.0001);");
			}
		}
		else if (sparseS) {
			const size_t nNZ = mxGetNumberOfElements(mxGetField(options, 0, "valL"));
			const size_t nCol = mxGetNumberOfElements(mxGetField(options, 0, "rowIndL"));
			const size_t nRow = mxGetNumberOfElements(mxGetField(options, 0, "colIndL"));
			if (DEBUG) {
				mexPrintf("nNZ = %u\n", nNZ);
				mexPrintf("nCol = %u\n", nCol);
				mexPrintf("nRow = %u\n", nRow);
			}
			const float* valuesL = loadFloats(options, "valL");
			const float* valuesLi = (float*)mxGetImagData(mxGetField(options, 0, "valLi"));
			const int32_t* colL = (int32_t*)mxGetData(mxGetField(options, 0, "colIndL"));
			rowL = (int32_t*)mxGetData(mxGetField(options, 0, "rowIndL"));
			if (complexRef && complexL)
				Lvalues = kompleksiset(nNZ, 1, valuesL, valuesLi);
			else
				Lvalues = array(nNZ, valuesL, afHost);
			Lcol = array(nNZ, 1, colL, afHost);
			lSize = imDim;
		}
		else {
			float* L = (float*)mxGetData(mxGetField(options, 0, "L"));
			if (complexRef && augType > 0) {
				float* Li = (float*)mxGetData(mxGetField(options, 0, "Li"));
				LL = kompleksiset(imDim, imDim, L, Li);
			}
			else {
				LL = array(imDim, imDim, L, afHost);
			}
		}
		if (regularization == 4 || (regularization == 3 && prior == 7)) {
			if (sparseL) {
				lCols.resize(imDim + 1, 0);
				std::transform(lCol, lCol + imDim + 1, std::begin(lCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
				Lrow = array(imDim + 1, 1, lCols.data(), afHost);
			}
			else
				Lrow = array(imDim + 1, 1, rowL, afHost);
			if (DEBUG) {
				mexPrintf("Lrow.dims(0) = %u\n", Lrow.dims(0));
				mexEvalString("pause(.0001);");
			}
			LL = sparse(imDim, imDim, real(Lvalues), Lrow, Lcol);
			if (complexRef && complexL)
				LLi = sparse(imDim, imDim, imag(Lvalues), Lrow, Lcol);
			lSize = 0ULL;
		}
		if (regularization == 1) {
			if (sparseL) {
				lCols.resize(imDim + 1, 0);
				std::transform(lCol, lCol + imDim + 1, std::begin(lCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
				Lrow = array(imDim + 1, 1, lCols.data(), afHost);
				Ly = sparse(imDim, imDim, real(Lvalues), Lrow, Lcol);
				if (complexRef && augType > 0 && complexL)
					Lyi = sparse(imDim, imDim, imag(Lvalues), Lrow, Lcol);
			}
			else if (sparseS) {
				Lrow = array(imDim + 1, 1, rowL, afHost);
				Ly = sparse(imDim, imDim, Lvalues, Lrow, Lcol);
				if (complexRef && augType > 0 && complexL)
					Lyi = sparse(imDim, imDim, imag(Lvalues), Lrow, Lcol);
			}
			if (augType == 0) {
				if (sparseL || sparseS)
					Ly = matmul(Ly, constant(1.f, imDim, 1));
				else
					Ly = matmul(LL, constant(1.f, imDim, 1));
			}
			else {
				array refImage = array(imDim, (float*)mxGetData(mxGetField(options, 0, "referenceImage")));
				if (sparseL || sparseS)
					Ly = matmul(Ly, refImage);
				else
					Ly = matmul(real(LL), refImage);
				if (complexRef && complexL) {
					array refImageI = array(imDim, (float*)mxGetImagData(mxGetField(options, imDim, "referenceImage")));
					if (sparseL || sparseS)
						Lyi = matmul(Lyi, refImageI);
					else
						Lyi = matmul(imag(LL), refImageI);
				}
				if (DEBUG) {
					mexPrintf("Ly.summa = %f\n", af::sum<float>(flat(Ly)));
					mexEvalString("pause(.0001);");
				}
			}
		}
	}
	if (sparseS) {
		const size_t nNZ = mxGetNzmax(mxGetField(options, 0, "H"));
		const size_t nCol = mxGetN(mxGetField(options, 0, "H"));
		const size_t nRow = mxGetM(mxGetField(options, 0, "H"));
		const double* SS = loadDoubles(options, "H");
		if (DEBUG) {
			mexPrintf("nNZ = %u\n", nNZ);
			mexPrintf("nCol = %u\n", nCol);
			mexPrintf("nRow = %u\n", nRow);
			mexEvalString("pause(.0001);");
		}
		S1.resize(nNZ, 0.f);
		std::transform(SS, SS + nNZ, std::begin(S1), [&](const double& value) { return static_cast<float>(value); });
		if (complexS && (complexType == 3 || complexType == 2)) {
			const double* SS2 = (double*)mxGetImagData(mxGetField(options, 0, "H"));
			S2.resize(nNZ, 0.f);
			std::transform(SS2, SS2 + nNZ, std::begin(S2), [&](const double& value) { return static_cast<float>(value); });
		}
		const size_t* sRow = reinterpret_cast<size_t*>(mxGetIr(mxGetField(options, 0, "H")));
		sRows.resize(nNZ, 0U);
		std::transform(sRow, sRow + nNZ, std::begin(sRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
		sCol = reinterpret_cast<size_t*>(mxGetJc(mxGetField(options, 0, "H")));
		sCols.resize((NmU * window + 1 + lSize) * hnU, 0);
		if (DEBUG) {
			mexPrintf("sCols.size() = %d\n", sCols.size());
			mexEvalString("pause(.0001);");
		}
		uint64_t uu = 0ULL;
		uint64_t ee = 1ULL;
		int32_t apu = 0;
		uint64_t ui = 0ULL;
		bool flip = true;
		for (uint64_t ii = 0ULL; ii < hnU; ii++) {
			uint64_t rr = 1ULL;
			sCols[uu] = 0;
			uu++;
			apu = static_cast<int32_t>(sCol[Nm * ii]);
			if (window > 1)
				ee = Nm * ii + 1ULL;
			for (uint64_t ll = 1ULL; ll <= (Nm * window + lSize); ll++) {
				if (ll > (Nm * window - (window - 1ULL)) && regularization == 1) {
					if (sparseL)
						sCols[uu] = sCols[ui] + static_cast<int32_t>(lCol[rr]);
					else
						sCols[uu] = sCols[ui] + rowL[rr];
					uu++;
					rr++;
				}
				else {
					if (ii >= (hnU - (window - 1))) {
						if (ll <= (hnU - ii) * Nm)
							sCols[uu] = static_cast<int32_t>(sCol[ee]) - apu;
						else {
							if (flip) {
								ee = 1ULL;
								flip = false;
							}
							sCols[uu] = static_cast<int32_t>(sCol[ee]);
						}
					}
					else
						sCols[uu] = static_cast<int32_t>(sCol[ee]) - apu;
					ui = uu;
					uu++;
					ee++;
				}
			}
			flip = true;
			//mexPrintf("ee = %d\n", ee);
			//mexPrintf("uu = %d\n", uu);
		}
		if (DEBUG) {
			mexPrintf("sCols.size() = %d\n", sCols.size());
		}
		if (storeData) {
			S.resize(hnU);
			Svalues = array(nNZ, S1.data(), afHost);
			if (complexS && (complexType == 3 || complexType == 2)) {
				Svaluesi = array(nNZ, S2.data(), afHost);
				Si.resize(hnU);
			}
			Scol = array(nNZ, 1, sRows.data(), afHost);
			Srow = array(sCols.size(), sCols.data(), afHost);
			if (DEBUG) {
				mexPrintf("Srow.dims(0) = %d\n", Srow.dims(0));
				mexPrintf("Scol.dims(0) = %d\n", Scol.dims(0));
				mexPrintf("Svalues.dims(0) = %d\n", Svalues.dims(0));
				mexPrintf("S.size() = %d\n", S.size());
				mexPrintf("Nm = %d\n", Nm);
				mexPrintf("imDim = %d\n", imDim);
				mexPrintf("(hnU - window + 1) = %d\n", (hnU - window + 1));
				mexPrintf("hnU = %d\n", hnU);
				mexEvalString("pause(.0001);");
			}
			for (uint64_t kk = 0ULL; kk < hnU; kk++) {
				if (window == 1 || (window > 1 && kk < (hnU - window + 1))) {
					if (regularization == 1)
						S[kk] = sparse(lSize + Nm * window, imDimN, join(0, Svalues(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Lvalues), Srow(seq((Nm * window + lSize) * kk + kk, (Nm * window + lSize) * (kk + 1ULL) + kk)), join(0, Scol(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Lcol));
					else
						S[kk] = sparse(Nm * window, imDimN, Svalues(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Srow(seq(Nm * window * kk + kk, Nm * window * (kk + 1ULL) + kk)), Scol(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)));
					if (complexS && (complexType == 3 || complexType == 2)) {
						if (regularization == 1)
							Si[kk] = sparse(lSize + Nm * window, imDimN, join(0, Svaluesi(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Lvalues), Srow(seq((Nm + lSize) * kk + kk, (Nm + lSize) * (kk + 1ULL) + kk + Nm * (window - 1ULL))), join(0, Scol(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Lcol));
						else
							Si[kk] = sparse(Nm * window, imDimN, Svaluesi(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Srow(seq(Nm * window * kk + kk, Nm * window * (kk + 1ULL) + kk)), Scol(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)));
					}
					//mexPrintf("kk = %d\n", kk);
				}
				else {
					if (DEBUG) {
						mexPrintf("Nm * (window - hnU - kk) = %d\n", Nm * (window - (hnU - kk)));
						mexPrintf("sCol[Nm * (window - hnU - kk)] = %d\n", sCol[Nm * (window - (hnU - kk))]);
						mexPrintf("sCol[Nm * kk] = %d\n", sCol[Nm * kk]);
						mexPrintf("kk = %d\n", kk);
						mexPrintf("Nm * window * (kk + 1ULL) + kk = %d\n", Nm * window * (kk + 1ULL) + kk);
						mexEvalString("pause(.0001);");
					}
					if (regularization == 1)
						S[kk] = sparse(lSize + Nm * window, imDimN, join(0, join(0, Svalues(seq(sCol[Nm * kk], end)), Svalues(seq(0, sCol[Nm * (window - (hnU - kk))]))), Lvalues), Srow(seq((Nm * window + lSize) * kk + kk, (Nm * window + lSize) * (kk + 1ULL) + kk)), join(0, join(0, Scol(seq(sCol[Nm * kk], end)), Scol(seq(0, sCol[Nm * (window - (hnU - kk))]))), Lcol));
					else
						S[kk] = sparse(Nm * window, imDimN, join(0, Svalues(seq(sCol[Nm * kk], end)), Svalues(seq(0, sCol[Nm * (window - (hnU - kk))]))), Srow(seq(Nm * window * kk + kk, Nm * window * (kk + 1ULL) + kk)), join(0, Scol(seq(sCol[Nm * kk], end)), Scol(seq(0, sCol[Nm * (window - (hnU - kk))]))));

					if (complexS && (complexType == 3 || complexType == 2)) {
						if (regularization == 1)
							Si[kk] = sparse(lSize + Nm * window, imDimN, join(0, Svaluesi(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Lvalues), Srow(seq((Nm + lSize) * kk + kk, (Nm + lSize) * (kk + 1ULL) + kk + Nm * (window - 1ULL))), join(0, Scol(seq(sCol[Nm * kk], sCol[Nm * (kk + 1) + Nm * (window - 1ULL)] - 1)), Lcol));
						else
							Si[kk] = sparse(Nm * window, imDimN, join(0, Svaluesi(seq(sCol[Nm * kk], end)), Svalues(seq(0, sCol[Nm * (window - (hnU - kk))]))), Srow(seq(Nm * window * kk + kk, Nm * window * (kk + 1ULL) + kk)), join(0, Scol(seq(sCol[Nm * kk], end)), Scol(seq(0, sCol[Nm * (window - (hnU - kk))]))));
					}
				}
				//mexPrintf("Nm * window * (kk + 1ULL) + kk = %d\n", Nm* window* (kk + 1ULL) + kk);
				//mexPrintf("kk = %d\n", kk);
			}
			if (DEBUG) {
				mexPrintf("S[0].dims(0) = %d\n", S[0].dims(0));
				mexPrintf("S[0].dims(1) = %d\n", S[0].dims(1));
				mexEvalString("pause(.0001);");
			}
		}
		else {
			Svalues = array(sCol[Nm + Nm * (window - 1ULL)], S1.data(), afHost);
			if (complexS && (complexType == 3 || complexType == 2))
				Svaluesi = array(sCol[Nm + Nm * (window - 1ULL)], S2.data(), afHost);
			Srow = array((Nm + lSize) + 1 + Nm * (window - 1ULL), sCols.data(), afHost);
			Scol = array(sCol[Nm + Nm * (window - 1ULL)], 1, sRows.data(), afHost);
			if (regularization == 1)
				S[0] = sparse(lSize + Nm * window, imDimN, join(0, Svalues(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)), Lvalues), Srow(seq(0, (Nm + lSize) + Nm * (window - 1ULL))), join(0, Scol(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)), Lcol));
			else
				S[0] = sparse(Nm * window, imDimN, Svalues(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)), Srow(seq(0, Nm * window)), Scol(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)));
			if (complexS && (complexType == 3 || complexType == 2)) {
				if (regularization == 1)
					Si[0] = sparse(lSize + Nm * window, imDimN, join(0, Svaluesi(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)), Lvalues), Srow(seq(0, (Nm + lSize) + Nm * (window - 1ULL))), join(0, Scol(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)), Lcol));
				else
					Si[0] = sparse(Nm * window, imDimN, Svaluesi(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)), Srow(seq(0, Nm * window)), Scol(seq(sCol[0], sCol[Nm + Nm * (window - 1ULL)] - 1)));
			}
			hnU = 1ULL;
		}
	}
	else {
		SS3 = (float*)mxGetData(mxGetField(options, 0, "H"));
		if (complexS && (complexType == 3 || complexType == 2)) {
			SS4 = (float*)mxGetImagData(mxGetField(options, 0, "H"));
		}
		if (storeData) {
			S.resize(hnU);
			if (complexS && (complexType == 3 || complexType == 2))
				Si.resize(hnU);
			for (uint64_t kk = 0ULL; kk < hnU; kk++) {
				if (window == 1 || (window > 1 && kk < (hnU - window + 1))) {
					if (regularization == 1) {
						if (sparseL) {
							S[kk] = join(0, transpose(array(imDimN, Nm * window, &SS3[kk * (Nm * imDimN)])), dense(LL));
							if (complexS && (complexType == 3 || complexType == 2))
								Si[kk] = join(0, transpose(array(imDimN, Nm * window, &SS4[kk * (Nm * imDimN)])), dense(LL));
						}
						else {
							S[kk] = join(0, transpose(array(imDimN, Nm * window, &SS3[kk * (Nm * imDimN)])), LL);
							if (complexS && (complexType == 3 || complexType == 2))
								Si[kk] = join(0, transpose(array(imDimN, Nm * window, &SS4[kk * (Nm * imDimN)])), LL);
						}
					}
					else {
						S[kk] = transpose(array(imDimN, Nm * window, &SS3[kk * (Nm * imDimN)]));
						if (complexS && (complexType == 3 || complexType == 2))
							Si[kk] = transpose(array(imDimN, Nm * window, &SS4[kk * (Nm * imDimN)]));
					}
				}
				else {
					if (regularization == 1) {
						if (sparseL) {
							S[kk] = join(0, join(0, transpose(array(imDimN, Nm * (hnU - kk), &SS3[kk * (Nm * imDimN)])), transpose(array(imDimN, Nm * (window - (hnU - kk)), &SS3[0]))), dense(LL));
							if (complexS && (complexType == 3 || complexType == 2))
								Si[kk] = join(0, join(0, transpose(array(imDimN, Nm * (hnU - kk), &SS4[kk * (Nm * imDimN)])), transpose(array(imDimN, Nm * (window - (hnU - kk)), &SS4[0]))), dense(LL));
						}
						else {
							S[kk] = join(0, join(0, transpose(array(imDimN, Nm * (hnU - kk), &SS3[kk * (Nm * imDimN)])), transpose(array(imDimN, Nm * (window - (hnU - kk)), &SS3[0]))), LL);
							if (complexS && (complexType == 3 || complexType == 2))
								Si[kk] = join(0, join(0, transpose(array(imDimN, Nm * (hnU - kk), &SS4[kk * (Nm * imDimN)])), transpose(array(imDimN, Nm * (window - (hnU - kk)), &SS4[0]))), LL);
						}
					}
					else {
						S[kk] = join(0, transpose(array(imDimN, Nm * (hnU - kk), &SS3[kk * (Nm * imDimN)])), transpose(array(imDimN, Nm * (window - (hnU - kk)), &SS3[0])));
						if (complexS && (complexType == 3 || complexType == 2))
							Si[kk] = join(0, transpose(array(imDimN, Nm * (hnU - kk), &SS4[kk * (Nm * imDimN)])), transpose(array(imDimN, Nm * (window - (hnU - kk)), &SS4[0])));
					}
				}
			}
		}
		else {
			if (complexS && (complexType == 3 || complexType == 2))
				Si.resize(1);
			if (regularization == 1) {
				if (sparseL) {
					S[0] = join(0, transpose(array(imDimN, Nm * window, SS3)), dense(LL));
					if (complexS && (complexType == 3 || complexType == 2))
						Si[0] = join(0, transpose(array(imDimN, Nm * window, SS4)), dense(LL));
				}
				else {
					S[0] = join(0, transpose(array(imDimN, Nm * window, SS3)), LL);
					if (complexS && (complexType == 3 || complexType == 2))
						Si[0] = join(0, transpose(array(imDimN, Nm * window, SS4)), LL);
				}
			}
			else {
				S[0] = transpose(array(imDimN, Nm * window, SS3));
				if (DEBUG) {
					mexPrintf("S[0].dims(0) = %d\n", S[0].dims(0));
					mexPrintf("S[0].dims(1) = %d\n", S[0].dims(1));
					mexEvalString("pause(.0001);");
				}
				if (complexS && (complexType == 3 || complexType == 2))
					Si[0] = transpose(array(imDimN, Nm * window, SS4));
				//mexPrintf("Si[0].dims(0) = %d\n", Si[0].dims(0));
				//mexPrintf("Si[0].dims(1) = %d\n", Si[0].dims(1));
				//mexEvalString("pause(.0001);");
			}
			hnU = 1ULL;
		}
	}


	bool sparseQ = (bool)mxGetScalar(mxGetField(options, 0, "sparseQ"));
	const bool tvQ = (bool)mxGetScalar(mxGetField(options, 0, "tvQ"));
	const bool complexQ = (bool)mxGetScalar(mxGetField(options, 0, "complexQ"));
	size_t sizeQ = 1ULL;
	if (tvQ) {
		const size_t nSQ = mxGetNumberOfElements(mxGetField(options, 0, "Q"));
		sizeQ = nSQ;
	}
	if (DEBUG) {
		mexPrintf("sizeQ = %u\n", sizeQ);
		mexEvalString("pause(.0001);");
	}
	std::vector<array> Q(sizeQ);
	std::vector<array> Qi;
	std::vector<array> Q2;
	std::vector<array> Q2i;
	if (complexType == 2)
		Qi.resize(sizeQ);
	if (algorithm == 10 && sparseQ) {
		Q2.resize(sizeQ);
		if (complexType == 2)
			Q2i.resize(sizeQ);
	}
	else if (algorithm == 10 && !sparseQ) {
		Q2.resize(1);
		if (complexType == 2)
			Q2i.resize(1);
	}
	if (!sparseQ) {
		const float* nwr = loadFloats(options, "Q");
		if ((complexType == 2 || complexType == 3 || complexType == 4) && complexQ) {
			const float* nwi = (float*)mxGetImagData(mxGetField(options, 0, "Q"));;
			for (uint64_t kk = 0ULL; kk < sizeQ; kk++) {
				if (complexType == 4)
					Q[kk] = kompleksiset(imDimN, DimZ, nwr + kk * imDimN * DimZ, nwi + kk * imDimN * DimZ);
				else if (complexType == 2) {
					Q[kk] = array(imDimN, DimZ, nwr + kk * imDimN * DimZ);
					Qi[kk] = array(imDimN, DimZ, nwi + kk * imDimN * DimZ);
				}
				else {
					Q[kk] = join(0, array(imDimN, DimZ, nwr + kk * imDimN * DimZ), array(imDimN, DimZ, nwi + kk * imDimN * DimZ));
				}
			}
		}
		else {
			for (uint64_t kk = 0ULL; kk < sizeQ; kk++) {
				//if (useKineticModel) {
				//	const float* Dval = loadFloats(options, "D");
				//	const int32_t* Drow = (int32_t*)mxGetData(mxGetField(options, 0, "Drow"));
				//	const int32_t* Dcol = (int32_t*)mxGetData(mxGetField(options, 0, "Dcol"));
				//	Q[kk] = sparse(imDimN, imDimN, imDimN, Dval + kk * imDimN, Drow, Dcol, f32, AF_STORAGE_CSR, afHost);
				//}
				//else
				Q[kk] = array(imDimN, DimZ, nwr + kk * imDimN * DimZ, afHost);
			}
		}
		//if (useKineticModel)
		//	sparseQ = true;
	}
	else {
		const size_t nCol = mxGetN(mxGetCell(mxGetField(options, 0, "Q"), 0));
		const size_t nRow = mxGetM(mxGetCell(mxGetField(options, 0, "Q"), 0));
		if (complexType == 2 || complexType == 3) {
			Qvalues.resize(sizeQ);
			if (complexType == 2)
				Qvaluesi.resize(sizeQ);
			Qrow.resize(sizeQ);
			Qcol.resize(sizeQ);
		}
		for (uint64_t tt = 0ULL; tt < sizeQ; tt++) {
			const size_t nNZ = mxGetNzmax(mxGetCell(mxGetField(options, 0, "Q"), tt));
			const double* QQ = (double*)mxGetData(mxGetCell(mxGetField(options, 0, "Q"), tt));
			if (DEBUG) {
				mexPrintf("nNZQ = %u\n", nNZ);
				mexPrintf("nColQ = %u\n", nCol);
				mexPrintf("nRowQ = %u\n", nRow);
				mexEvalString("pause(.0001);");
			}
			std::vector<float> Q1(nNZ);
			std::vector<float> Q22;
			std::transform(QQ, QQ + nNZ, std::begin(Q1), [&](const double& value) { return static_cast<float>(value); });
			if ((complexType == 2 || complexType == 3) && complexQ) {
				const double* QQ2 = (double*)mxGetImagData(mxGetCell(mxGetField(options, 0, "Q"), tt));
				Q22.resize(nNZ, 0.f);
				std::transform(QQ2, QQ2 + nNZ, std::begin(Q22), [&](const double& value) { return static_cast<float>(value); });
			}
			float summa = 0.f;
			for (int uu = 0; uu < nNZ; uu++)
				summa += Q1[uu];
			const size_t* qRow = reinterpret_cast<size_t*>(mxGetIr(mxGetCell(mxGetField(options, 0, "Q"), tt)));
			std::vector<int32_t> qRows(nNZ, 0);
			std::transform(qRow, qRow + nNZ, std::begin(qRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
			std::vector<int32_t> qCols(nCol + 1, 0);
			const size_t* qCol = reinterpret_cast<size_t*>(mxGetJc(mxGetCell(mxGetField(options, 0, "Q"), tt)));
			std::transform(qCol, qCol + nCol + 1, std::begin(qCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
			if (DEBUG) {
				mexPrintf("summa = %f\n", summa);
				mexPrintf("qCols.size() = %u\n", qCols.size());
				mexEvalString("pause(.0001);");
			}
			if ((complexType == 2 || complexType == 3) && complexQ) {
				if (complexType == 3) {
					Qvalues[tt] = join(0, array(nNZ, 1, Q1.data()), array(nNZ, 1, Q22.data()));
					Qrow[tt] = array(qCols.size(), qCols.data());
					Qrow[tt] = join(0, Qrow[tt], Qrow[tt](seq(1, end)) + tile(Qrow[tt](end), Qrow[tt].dims(0) - 1));
					Qcol[tt] = array(qRows.size(), qRows.data());
					Qcol[tt] = join(0, Qcol[tt], Qcol[tt] + static_cast<int32_t>(nRow));
				}
				else {
					Qvalues[tt] = array(nNZ, 1, Q1.data());
					Qrow[tt] = array(qCols.size(), qCols.data());
					Qcol[tt] = array(qRows.size(), qRows.data());
				}
				Q[tt] = sparse(imDimU, imDimU, Qvalues[tt], Qrow[tt], Qcol[tt]);
				if (complexType == 2) {
					Qvaluesi[tt] = array(nNZ, 1, Q22.data());
					Qi[tt] = sparse(imDimN, imDimN, Qvaluesi[tt], Qrow[tt], Qcol[tt]);
				}
			}
			else
				Q[tt] = sparse(nCol, nRow, nNZ, Q1.data(), qCols.data(), qRows.data(), f32, AF_STORAGE_CSR, afHost);
		}
		if (DEBUG) {
			mexPrintf("Q[0].dims(0) = %d\n", Q[0].dims(0));
			mexPrintf("Q[0].dims(1) = %d\n", Q[0].dims(1));
			mexEvalString("pause(.0001);");
		}
		if (algorithm == 10 && useSmoother) {
			if (complexType == 2 || complexType == 3) {
				Qvalues2.resize(sizeQ);
				if (complexType == 2)
					Qvaluesi2.resize(sizeQ);
				Qrow2.resize(sizeQ);
				Qcol2.resize(sizeQ);
			}
			for (uint64_t tt = 0ULL; tt < sizeQ; tt++) {
				const size_t nNZ = mxGetNzmax(mxGetCell(mxGetField(options, 0, "Q2"), tt));
				const double* QQ = (double*)mxGetData(mxGetCell(mxGetField(options, 0, "Q2"), tt));
				if (DEBUG) {
					mexPrintf("nNZQ = %u\n", nNZ);
					mexPrintf("nColQ = %u\n", nCol);
					mexPrintf("nRowQ = %u\n", nRow);
					mexEvalString("pause(.0001);");
				}
				std::vector<float> Q1(nNZ);
				std::vector<float> Q22;
				std::transform(QQ, QQ + nNZ, std::begin(Q1), [&](const double& value) { return static_cast<float>(value); });
				if ((complexType == 2 || complexType == 3) && complexQ) {
					const double* QQ2 = (double*)mxGetImagData(mxGetCell(mxGetField(options, 0, "Q2"), tt));
					Q22.resize(nNZ, 0.f);
					std::transform(QQ2, QQ2 + nNZ, std::begin(Q22), [&](const double& value) { return static_cast<float>(value); });
				}
				const size_t* qRow = reinterpret_cast<size_t*>(mxGetIr(mxGetCell(mxGetField(options, 0, "Q2"), tt)));
				std::vector<int32_t> qRows(nNZ, 0);
				std::transform(qRow, qRow + nNZ, std::begin(qRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
				std::vector<int32_t> qCols(nCol + 1, 0);
				const size_t* qCol = reinterpret_cast<size_t*>(mxGetJc(mxGetCell(mxGetField(options, 0, "Q2"), tt)));
				std::transform(qCol, qCol + nCol + 1, std::begin(qCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
				if (DEBUG) {
					mexPrintf("qCols.size() = %u\n", qCols.size());
					mexEvalString("pause(.0001);");
				}
				if ((complexType == 2 || complexType == 3) && complexQ) {
					if (complexType == 3) {
						Qvalues2[tt] = join(0, array(nNZ, 1, Q1.data()), array(nNZ, 1, Q22.data()));
						Qrow2[tt] = array(qCols.size(), qCols.data());
						Qrow2[tt] = join(0, Qrow2[tt], Qrow2[tt](seq(1, end)) + tile(Qrow2[tt](end), Qrow2[tt].dims(0) - 1));
						Qcol2[tt] = array(qRows.size(), qRows.data());
						Qcol2[tt] = join(0, Qcol2[tt], Qcol2[tt] + static_cast<int32_t>(nRow));
					}
					else {
						Qvalues2[tt] = array(nNZ, 1, Q1.data());
						Qrow2[tt] = array(qCols.size(), qCols.data());
						Qcol2[tt] = array(qRows.size(), qRows.data());
					}
					Q2[tt] = sparse(imDimU, imDimU, Qvalues2[tt], Qrow2[tt], Qcol2[tt]);
					if (complexType == 2) {
						Qvaluesi2[tt] = array(nNZ, 1, Q22.data());
						Q2i[tt] = sparse(imDimN, imDimN, Qvaluesi2[tt], Qrow2[tt], Qcol2[tt]);
					}
				}
				else
					Q[tt] = sparse(imDimU, imDimU, nNZ, Q1.data(), qCols.data(), qRows.data(), f32, AF_STORAGE_CSR, afHost);
			}
		}
	}

	const bool sparseR = (bool)mxGetScalar(mxGetField(options, 0, "sparseR"));
	const bool tvR = (bool)mxGetScalar(mxGetField(options, 0, "tvR"));
	size_t sizeR = 1ULL;
	if (tvR)
		sizeR = Nt - (window - 1ULL);
	std::vector<array> R(sizeR);
	std::vector<array> Ri;
	std::vector<array> R2;
	std::vector<array> R2i;
	if (complexType == 2)
		Ri.resize(sizeR);
	if (algorithm == 5 || algorithm == 6) {
		R2.resize(sizeR);
		if (complexType == 2)
			R2i.resize(sizeR);
	}
	if (DEBUG) {
		mexPrintf("sizeR = %u\n", sizeR);
		mexEvalString("pause(.0001);");
	}
	if (!sparseR) {
		const float* nvr = loadFloats(options, "R");
		if (complexType == 2 || complexType == 3 || complexType == 4) {
			const float* nvi = (float*)mxGetImagData(mxGetField(options, 0, "R"));
			for (uint64_t kk = 0ULL; kk < sizeR; kk++) {
				if (complexType == 4)
					R[kk] = kompleksiset(Nm * window, 1, nvr + kk * Nm, nvi + kk * Nm);
				else if (complexType == 2) {
					R[kk] = array(Nm * window, 1, nvr + kk * Nm);
					Ri[kk] = array(Nm * window, 1, nvi + kk * Nm);
				}
				else {
					R[kk] = join(0, array(Nm * window, 1, nvr + kk * Nm), array(Nm * window, 1, nvi + kk * Nm));
				}
			}
		}
		else {
			for (uint64_t kk = 0ULL; kk < sizeR; kk++)
				R[kk] = array(Nm * window, 1, nvr + kk * Nm, afHost);
		}
		//if (algorithm == 5 || algorithm == 6) {
		//	const float* nvr2 = loadFloats(options, "R2");
		//	if (complexType == 2 || complexType == 3 || complexType == 4) {
		//		const float* nvi2 = (float*)mxGetImagData(mxGetField(options, 0, "R2"));
		//		for (uint64_t kk = 0ULL; kk < sizeR; kk++) {
		//			if (complexType == 4)
		//				R2[kk] = kompleksiset(Nm * window, 1, nvr2 + kk * Nm, nvi2 + kk * Nm);
		//			else if (complexType == 2) {
		//				R2[kk] = array(Nm * window, 1, nvr2 + kk * Nm);
		//				R2i[kk] = array(Nm * window, 1, nvi2 + kk * Nm);
		//			}
		//			else {
		//				R2[kk] = join(0, array(Nm * window, 1, nvr2 + kk * Nm), array(Nm * window, 1, nvi2 + kk * Nm));
		//			}
		//		}
		//	}
		//	else {
		//		for (uint64_t kk = 0ULL; kk < sizeR; kk++)
		//			R2[kk] = array(Nm * window, 1, nvr2 + kk * Nm, afHost);
		//	}
		//}
	}
	else {
		const size_t nCol = mxGetN(mxGetCell(mxGetField(options, 0, "R"), 0));
		const size_t nRow = mxGetM(mxGetCell(mxGetField(options, 0, "R"), 0));
		if (complexType == 2 || complexType == 3) {
			Rvalues.resize(sizeR);
			if (complexType == 2)
				Rvaluesi.resize(sizeR);
			Rrow.resize(sizeR);
			Rcol.resize(sizeR);
		}
		for (uint64_t tt = 0ULL; tt < sizeR; tt++) {
			const size_t nNZ = mxGetNzmax(mxGetCell(mxGetField(options, 0, "R"), tt));
			const double* RR = (double*)mxGetData(mxGetCell(mxGetField(options, 0, "R"), tt));
			if (DEBUG) {
				mexPrintf("nNZR = %u\n", nNZ);
				mexPrintf("nColR = %u\n", nCol);
				mexPrintf("nRowR = %u\n", nRow);
				mexEvalString("pause(.0001);");
			}
			std::vector<float> R1(nNZ);
			std::vector<float> R12;
			std::transform(RR, RR + nNZ, std::begin(R1), [&](const double& value) { return static_cast<float>(value); });
			if (complexType == 2 || complexType == 3) {
				const double* RR2 = (double*)mxGetImagData(mxGetCell(mxGetField(options, 0, "R"), tt));
				R12.resize(nNZ, 0.f);
				std::transform(RR2, RR2 + nNZ, std::begin(R12), [&](const double& value) { return static_cast<float>(value); });
			}
			const size_t* rRow = reinterpret_cast<size_t*>(mxGetIr(mxGetCell(mxGetField(options, 0, "R"), tt)));
			std::vector<int32_t> rRows(nNZ, 0U);
			std::transform(rRow, rRow + nNZ, std::begin(rRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
			std::vector<int32_t> rCols(nCol + 1, 0U);
			const size_t* rCol = reinterpret_cast<size_t*>(mxGetJc(mxGetCell(mxGetField(options, 0, "R"), tt)));
			std::transform(rCol, rCol + nCol + 1, std::begin(rCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
			if (complexType == 2 || complexType == 3) {
				if (complexType == 3) {
					Rvalues[tt] = join(0, array(nNZ, 1, R1.data()), array(nNZ, 1, R12.data()));
					Rrow[tt] = array(rCols.size(), rCols.data());
					Rrow[tt] = join(0, Rrow[tt], Rrow[tt](seq(1, end)) + tile(Rrow[tt](end), Rrow[tt].dims(0) - 1), 1);
					Rcol[tt] = array(rRows.size(), rRows.data());
					Rcol[tt] = join(0, Rcol[tt], Rcol[tt] + static_cast<int32_t>(nRow));
				}
				else {
					Rvalues[tt] = array(nNZ, 1, R1.data());
					Rrow[tt] = array(rCols.size(), rCols.data());
					Rcol[tt] = array(rRows.size(), rRows.data());
				}
				R[tt] = sparse(nCol, nRow, Rvalues[tt], Rrow[tt], Rcol[tt]);
				if (complexType == 2) {
					Rvaluesi[tt] = array(nNZ, 1, R12.data());
					Ri[tt] = sparse(nCol, nRow, Rvaluesi[tt], Rrow[tt], Rcol[tt]);
				}
			}
			else
				R[tt] = sparse(nCol, nRow, nNZ, R1.data(), rCols.data(), rRows.data(), f32, AF_STORAGE_CSR, afHost);
		}
		if (algorithm == 5 || algorithm == 6) {
			if (complexType == 2 || complexType == 3) {
				Rvalues2.resize(sizeR);
				if (complexType == 2)
					Rvaluesi2.resize(sizeR);
				Rrow2.resize(sizeR);
				Rcol2.resize(sizeR);
			}
			for (uint64_t tt = 0ULL; tt < sizeR; tt++) {
				const size_t nNZ = mxGetNzmax(mxGetCell(mxGetField(options, 0, "R2"), tt));
				const double* RR = (double*)mxGetData(mxGetCell(mxGetField(options, 0, "R2"), tt));
				if (DEBUG) {
					mexPrintf("nNZR = %u\n", nNZ);
					mexPrintf("nColR = %u\n", nCol);
					mexPrintf("nRowR = %u\n", nRow);
					mexEvalString("pause(.0001);");
				}
				std::vector<float> R1(nNZ);
				std::vector<float> R12;
				std::transform(RR, RR + nNZ, std::begin(R1), [&](const double& value) { return static_cast<float>(value); });
				if (complexType == 2 || complexType == 3) {
					const double* RR2 = (double*)mxGetImagData(mxGetCell(mxGetField(options, 0, "R2"), tt));
					R12.resize(nNZ, 0.f);
					std::transform(RR2, RR2 + nNZ, std::begin(R12), [&](const double& value) { return static_cast<float>(value); });
				}
				const size_t* rRow = reinterpret_cast<size_t*>(mxGetIr(mxGetCell(mxGetField(options, 0, "R2"), tt)));
				std::vector<int32_t> rRows(nNZ, 0U);
				std::transform(rRow, rRow + nNZ, std::begin(rRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
				std::vector<int32_t> rCols(nCol + 1, 0U);
				const size_t* rCol = reinterpret_cast<size_t*>(mxGetJc(mxGetCell(mxGetField(options, 0, "R2"), tt)));
				std::transform(rCol, rCol + nCol + 1, std::begin(rCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
				if (complexType == 2 || complexType == 3) {
					if (complexType == 3) {
						Rvalues2[tt] = join(0, array(nNZ, 1, R1.data()), array(nNZ, 1, R12.data()));
						Rrow2[tt] = array(rCols.size(), rCols.data());
						Rrow2[tt] = join(0, Rrow2[tt], Rrow2[tt](seq(1, end)) + tile(Rrow2[tt](end), Rrow2[tt].dims(0) - 1), 1);
						Rcol2[tt] = array(rRows.size(), rRows.data());
						Rcol2[tt] = join(0, Rcol2[tt], Rcol2[tt] + static_cast<int32_t>(nRow));
					}
					else {
						Rvalues2[tt] = array(nNZ, 1, R1.data());
						Rrow2[tt] = array(rCols.size(), rCols.data());
						Rcol2[tt] = array(rRows.size(), rRows.data());
					}
					R2[tt] = sparse(nCol, nRow, Rvalues2[tt], Rrow2[tt], Rcol2[tt]);
					if (complexType == 2) {
						Rvaluesi2[tt] = array(nNZ, 1, R12.data());
						R2i[tt] = sparse(nCol, nRow, Rvaluesi2[tt], Rrow2[tt], Rcol2[tt]);
					}
				}
				else
					R2[tt] = sparse(nCol, nRow, nNZ, R1.data(), rCols.data(), rRows.data(), f32, AF_STORAGE_CSR, afHost);
			}

		}
	}

	array Pplus, Pplusi;
	if (algorithm <= 2 || algorithm == 9) {
		const float* nfr = (float*)mxGetData(mxGetField(options, 0, "P0"));
		const size_t nRowP = mxGetNumberOfElements(mxGetField(options, 0, "P0"));
		if (DEBUG) {
			mexPrintf("nRowP = %d\n", nRowP);
			mexEvalString("pause(.0001);");
		}
		if (complexType == 2 || complexType == 3) {
			const float* nfi = (float*)mxGetImagData(mxGetField(options, 0, "P0"));
			if (complexType == 2) {
				Pplus = diag(array(nRowP, 1, nfr, afHost), 0, false);
				Pplusi = diag(array(nRowP, 1, nfi, afHost), 0, false);
			}
			else
				Pplus = diag(join(0, array(nRowP, 1, nfr, afHost), array(nRowP, 1, nfi, afHost)), 0, false);
		}
		else if (complexType == 4) {
			const float* nfi = (float*)mxGetImagData(mxGetField(options, 0, "P0"));
			Pplus = diag(kompleksiset(nRowP, 1, nfr, nfi), 0, false);
		}
		else
			if (algorithm == 9) {
				if (useKineticModel)
					Pplus = diag(join(0, array(nRowP, 1, nfr, afHost), array(nRowP, 1, nfr, afHost)), 0, false);
				else
					Pplus = diag(array(nRowP, 1, nfr, afHost), 0, false);
			}
			else
				Pplus = diag(array(nRowP, 1, nfr, afHost), 0, false);
		if (DEBUG) {
			mexPrintf("Pplus.dims(0) = %d\n", Pplus.dims(0));
			mexPrintf("Pplus.dims(1) = %d\n", Pplus.dims(1));
			mexEvalString("pause(.0001);");
		}
	}
	else if (algorithm == 10 || algorithm == 11) {
		const size_t nRowP = mxGetM(mxGetField(options, 0, "P0"));
		const size_t nColP = mxGetN(mxGetField(options, 0, "P0"));
		if (DEBUG) {
			mexPrintf("nRowP = %d\n", nRowP);
			mexEvalString("pause(.0001);");
		}
		if (complexType <= 2) {
			Pplus = array(nRowP, nColP, (float*)mxGetData(mxGetField(options, 0, "P0")), afHost);
			if ((algorithm == 11 && complexType == 1) || complexType == 2)
				Pplusi = array(nRowP, nColP, (float*)mxGetImagData(mxGetField(options, 0, "P0")), afHost);
		}
		else
			Pplus = join(0, array(nRowP, nColP, (float*)mxGetData(mxGetField(options, 0, "P0")), afHost), array(nRowP, nColP, (float*)mxGetImagData(mxGetField(options, 0, "P0")), afHost));
	}
	else if (algorithm == 12) {
		const float* nfr = (float*)mxGetData(mxGetField(options, 0, "P0"));
		const size_t nRowP = mxGetNumberOfElements(mxGetField(options, 0, "P0"));
		if (complexType == 2 || complexType == 3) {
			const float* nfi = (float*)mxGetImagData(mxGetField(options, 0, "P0"));
			if (complexType == 2) {
				Pplus = tile(array(nRowP, 1, nfr, afHost), 1, Nm);
				Pplusi = tile(array(nRowP, 1, nfi, afHost), 1, Nm);
			}
			else
				Pplus = tile(join(0, array(nRowP, 1, nfr, afHost), array(nRowP, 1, nfi, afHost)), 1, Nm);
		}
		else
			Pplus = tile(array(nRowP, 1, nfr, afHost), 1, Nm);
	}


	const bool useF = (bool)mxGetScalar(mxGetField(options, 0, "useF"));
	const bool sparseF = (bool)mxGetScalar(mxGetField(options, 0, "sparseF"));
	const bool tvF = (bool)mxGetScalar(mxGetField(options, 0, "tvF"));
	const bool complexF = (bool)mxGetScalar(mxGetField(options, 0, "complexF"));
	std::vector<array> F;
	uint64_t sizeF = 1ULL;
	std::vector<array> Fi;
	std::vector<array> Fvalues, Fvaluesi, Frow, Fcol;
	if (DEBUG) {
		mexPrintf("sparseF = %u\n", sparseF);
		mexEvalString("pause(.0001);");
	}
	if (useF || useKineticModel) {
		if (useKineticModel) {
			const float* Fval = loadFloats(options, "F");
			const size_t nNZF = mxGetNumberOfElements(mxGetField(options, 0, "F"));
			const int32_t* Frow = (int32_t*)mxGetData(mxGetField(options, 0, "Frow"));
			const size_t nRowF = mxGetNumberOfElements(mxGetField(options, 0, "Frow"));
			const int32_t* Fcol = (int32_t*)mxGetData(mxGetField(options, 0, "Fcol"));
			if (algorithm >= 1) {
				F.push_back(sparse(nRowF - 1ULL, nRowF - 1ULL, nNZF, Fval, Frow, Fcol, f32, AF_STORAGE_CSR, afHost));
			}
			else
				F.push_back(sparse(imDim, imDimU, nNZF, Fval, Frow, Fcol, f32, AF_STORAGE_CSR, afHost));
			Fi.resize(sizeF);
		}
		else {
			sizeF = mxGetNumberOfElements(mxGetField(options, 0, "F"));
			F.resize(sizeF);
			//if (complexType == 2)
			Fi.resize(sizeF);
			if (DEBUG) {
				mexPrintf("sizeF = %u\n", sizeF);
				mexEvalString("pause(.0001);");
			}
			if (sparseF) {
				const size_t nCol = mxGetN(mxGetCell(mxGetField(options, 0, "F"), 0));
				const size_t nRow = mxGetM(mxGetCell(mxGetField(options, 0, "F"), 0));
				if ((complexType == 2 || complexType == 3) && complexF) {
					Fvalues.resize(sizeF);
					if (complexType == 2)
						Fvaluesi.resize(sizeF);
					Frow.resize(sizeF);
					Fcol.resize(sizeF);
				}
				for (uint64_t tt = 0ULL; tt < sizeF; tt++) {
					const size_t nNZ = mxGetNzmax(mxGetCell(mxGetField(options, 0, "F"), tt));
					const double* FF = (double*)mxGetData(mxGetCell(mxGetField(options, 0, "F"), tt));
					if (DEBUG) {
						mexPrintf("nNZF = %u\n", nNZ);
						mexPrintf("nColF = %u\n", nCol);
						mexPrintf("nRowF = %u\n", nRow);
						mexEvalString("pause(.0001);");
					}
					std::vector<float> F1(nNZ, 0.f);
					std::vector<float> F2;
					std::transform(FF, FF + nNZ, std::begin(F1), [&](const double& value) { return static_cast<float>(value); });
					if ((complexType == 2 || complexType == 3) && complexF) {
						const double* FF2 = (double*)mxGetImagData(mxGetCell(mxGetField(options, 0, "F"), tt));
						F2.resize(nNZ, 0.f);
						std::transform(FF2, FF2 + nNZ, std::begin(F2), [&](const double& value) { return static_cast<float>(value); });
					}
					const size_t* fRow = reinterpret_cast<size_t*>(mxGetIr(mxGetCell(mxGetField(options, 0, "F"), tt)));
					std::vector<int32_t> fRows(nNZ, 0U);
					std::transform(fRow, fRow + nNZ, std::begin(fRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
					std::vector<int32_t> fCols(nCol + 1, 0U);
					const size_t* fCol = reinterpret_cast<size_t*>(mxGetJc(mxGetCell(mxGetField(options, 0, "F"), 0)));
					std::transform(fCol, fCol + nCol + 1, std::begin(fCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
					if ((complexType == 2 || complexType == 3) && complexF) {
						Fvalues[tt] = array(nNZ, 1, F1.data());
						Frow[tt] = array(fCols.size(), fCols.data());
						Fcol[tt] = array(fRows.size(), fRows.data());
						F[tt] = sparse(nCol, nRow, Fvalues[tt], Frow[tt], Fcol[tt]);
						if (complexType == 2) {
							Fvaluesi[tt] = array(nNZ, 1, F2.data());
							Fi[tt] = sparse(nCol, nRow, Fvaluesi[tt], Frow[tt], Fcol[tt]);
						}
					}
					else
						F[tt] = sparse(nCol, nRow, nNZ, F1.data(), fCols.data(), fRows.data(), f32, AF_STORAGE_CSR, afHost);
				}
			}
			else {
				for (uint64_t kk = 0ULL; kk < sizeF; kk++) {
					const float* FF = (float*)mxGetData(mxGetCell(mxGetField(options, 0, "F"), kk));
					const size_t nRow = mxGetM(mxGetCell(mxGetField(options, 0, "F"), 0));
					const size_t nCol = mxGetN(mxGetCell(mxGetField(options, 0, "F"), 0));
					if (complexF) {
						const float* FFi = (float*)mxGetImagData(mxGetCell(mxGetField(options, 0, "F"), kk));
						if (complexType == 4)
							F[kk] = kompleksiset(imDimU, imDim, FF, FFi);
						else {
							F[kk] = array(nRow, nCol, FF);
							Fi[kk] = array(nRow, nCol, FFi);
						}
						eval(F[kk]);
					}
					else {
						F[kk] = array(nRow, nCol, FF, afHost);
					}
				}
			}
		}
	}

	const bool useU = (bool)mxGetScalar(mxGetField(options, 0, "useU"));
	const bool tvU = (bool)mxGetScalar(mxGetField(options, 0, "tvU"));
	const bool complexU = (bool)mxGetScalar(mxGetField(options, 0, "complexU"));
	size_t sizeU = 1ULL;
	array u, ui;
	if (useU) {
		sizeU = mxGetN(mxGetField(options, 0, "u"));
		const float* uu = (float*)mxGetData(mxGetField(options, 0, "u"));
		if (complexU) {
			const float* uui = (float*)mxGetImagData(mxGetField(options, 0, "u"));
			u = kompleksiset(imDimN, sizeU, uu, uui);
			eval(u);
		}
		else {
			u = array(imDimN, sizeU, uu);
		}
	}
	if (DEBUG) {
		mexPrintf("useU = %u\n", useU);
		mexEvalString("pause(.0001);");
	}

	const bool useG = (bool)mxGetScalar(mxGetField(options, 0, "useG"));
	const bool sparseG = (bool)mxGetScalar(mxGetField(options, 0, "sparseG"));
	const bool tvG = (bool)mxGetScalar(mxGetField(options, 0, "tvG"));
	const bool complexG = (bool)mxGetScalar(mxGetField(options, 0, "complexG"));
	std::vector<array> G;
	uint64_t sizeG = 1ULL;
	std::vector<array> Gi;
	std::vector<array> Gvalues, Gvaluesi, Grow, Gcol;
	if (useG) {
		sizeG = mxGetNumberOfElements(mxGetField(options, 0, "G"));
		G.resize(sizeG);
		if (complexG)
			Gi.resize(sizeG);
		if (sparseG) {
			const size_t nCol = mxGetN(mxGetCell(mxGetField(options, 0, "G"), 0));
			const size_t nRow = mxGetM(mxGetCell(mxGetField(options, 0, "G"), 0));
			if ((complexType == 2 || complexType == 3) && complexG) {
				Gvalues.resize(sizeF);
				if (complexType == 2)
					Gvaluesi.resize(sizeF);
				Grow.resize(sizeF);
				Gcol.resize(sizeF);
			}
			for (uint64_t tt = 0ULL; tt < sizeG; tt++) {
				const size_t nNZ = mxGetNzmax(mxGetCell(mxGetField(options, 0, "G"), tt));
				const double* GG = (double*)mxGetData(mxGetCell(mxGetField(options, 0, "G"), tt));
				if (DEBUG) {
					mexPrintf("nNZ = %u\n", nNZ);
					mexPrintf("nCol = %u\n", nCol);
					mexPrintf("nRow = %u\n", nRow);
				}
				std::vector<float> G1(nNZ);
				std::vector<float> G2;
				std::transform(GG, GG + nNZ, std::begin(G1), [&](const double& value) { return static_cast<float>(value); });
				if ((complexType == 2 || complexType == 3) && complexG) {
					const double* GG2 = (double*)mxGetImagData(mxGetCell(mxGetField(options, 0, "G"), tt));
					G2.resize(nNZ, 0.f);
					std::transform(GG2, GG2 + nNZ, std::begin(G2), [&](const double& value) { return static_cast<float>(value); });
				}
				const size_t* gRow = reinterpret_cast<size_t*>(mxGetIr(mxGetCell(mxGetField(options, 0, "G"), tt)));
				std::vector<int32_t> gRows(nNZ, 0U);
				std::transform(gRow, gRow + nNZ, std::begin(gRows), [&](const size_t& value) { return static_cast<int32_t>(value); });
				std::vector<int32_t> gCols(nCol + 1, 0U);
				const size_t* gCol = reinterpret_cast<size_t*>(mxGetJc(mxGetCell(mxGetField(options, 0, "G"), 0)));
				std::transform(gCol, gCol + nCol + 1, std::begin(gCols), [&](const size_t& value) { return static_cast<int32_t>(value); });
				if ((complexType == 2 || complexType == 3) && complexG) {
					Gvalues[tt] = array(nNZ, 1, G1.data());
					Grow[tt] = array(gCols.size(), gCols.data());
					Gcol[tt] = array(gRows.size(), gRows.data());
					G[tt] = sparse(nCol, nRow, Gvalues[tt], Grow[tt], Gcol[tt]);
					if (complexType == 2) {
						Gvaluesi[tt] = array(nNZ, 1, G2.data());
						Gi[tt] = sparse(nCol, nRow, Gvaluesi[tt], Grow[tt], Gcol[tt]);
					}
				}
				else
					G[tt] = sparse(nCol, nRow, nNZ, G1.data(), gCols.data(), gRows.data(), f32, AF_STORAGE_CSR, afHost);
			}
		}
		else {
			for (uint64_t kk = 0ULL; kk < sizeF; kk++) {
				const float* GG = (float*)mxGetData(mxGetCell(mxGetField(options, 0, "G"), kk));
				if (complexG) {
					const float* GGi = (float*)mxGetImagData(mxGetCell(mxGetField(options, 0, "G"), kk));
					if (complexType == 4)
						G[kk] = kompleksiset(imDimN, imDimN, GG, GGi);
					else if (complexType == 3)
						G[kk] = join(0, join(1, array(imDimN, sizeU, GG), -1.f * array(imDimN, sizeU, GGi)), join(1, array(imDimN, sizeU, GGi), array(imDimN, sizeU, GG)));
					else {
						G[kk] = array(imDimN, imDimN, GG);
						Gi[kk] = array(imDimN, imDimN, GGi);
					}
				}
				else {
					G[kk] = array(imDimN, imDimN, GG, afHost);
				}
			}
		}
	}
	array Sigma, XX, YY, ZZ, Ured, Sred, V, PredApu, Pred;
	if (algorithm == 9) {
		//const bool useCustomCov = (uint64_t)mxGetScalar(mxGetField(options, 0, "useCustomCov"));
		const uint64_t covDimX = (uint64_t)mxGetScalar(mxGetField(options, 0, "covDimX"));
		const uint64_t covDimY = (uint64_t)mxGetScalar(mxGetField(options, 0, "covDimY"));
		const uint64_t covDimZ = (uint64_t)mxGetScalar(mxGetField(options, 0, "covDimZ"));
		if (DEBUG) {
			mexPrintf("covDimX = %d\n", covDimX);
			mexPrintf("covDimY = %d\n", covDimY);
			mexEvalString("pause(.0001);");
		}
		if (covDimZ == 1) {
			const size_t nCol = mxGetN(mxGetField(options, 0, "Sigma"));
			const size_t nRow = mxGetM(mxGetField(options, 0, "Sigma"));
			const size_t nEle = mxGetNumberOfElements(mxGetField(options, 0, "Sigma"));
			const size_t nSlice = (nCol * nRow) / nEle;
			Sigma = array(nRow, nCol, nSlice, (float*)mxGetData(mxGetField(options, 0, "Sigma")));
			af::svd(Ured, Sred, V, Sigma);
			if (DEBUG) {
				mexPrintf("Ured.dims(0) = %d\n", Ured.dims(0));
				mexPrintf("Ured.dims(1) = %d\n", Ured.dims(1));
				mexEvalString("pause(.0001);");
			}
			const uint64_t nBasis = (uint64_t)mxGetScalar(mxGetField(options, 0, "reducedBasisN"));
			XX = array(Nx, Ny, (float*)mxGetData(mxGetField(options, 0, "XX")));
			YY = array(Nx, Ny, (float*)mxGetData(mxGetField(options, 0, "YY")));
			PredApu = moddims(tile(sqrt(Sred(seq(0, nBasis)).T()), Ured.dims(0), 1) * Ured(span, seq(0, nBasis)), covDimX, covDimY, nBasis + 1);
			if (DEBUG) {
				mexPrintf("PredApu.dims(0) = %d\n", PredApu.dims(0));
				mexPrintf("PredApu.dims(1) = %d\n", PredApu.dims(1));
				mexEvalString("pause(.0001);");
			}
			//array testi = approx2(PredApu(span, 0), XX, YY);
			//mexPrintf("testi.dims(0) = %d\n", testi.dims(0));
			//mexPrintf("testi.dims(1) = %d\n", testi.dims(1));
			//mexEvalString("pause(.0001);");
			Pred = constant(0.f, Nx, Ny, nBasis + 1ULL);
			if (DEBUG) {
				mexPrintf("Pred.dims(0) = %d\n", Pred.dims(0));
				mexPrintf("Pred.dims(1) = %d\n", Pred.dims(1));
				mexEvalString("pause(.0001);");
			}
			af::sync();
			//gfor(seq i, nBasis + 1)
			for (int64_t i = 0; i <= nBasis; i++)
				Pred(span, span, i) = approx2(PredApu(span, span, i), XX, YY);
			if (DEBUG) {
				mexPrintf("Pred.dims(0) = %d\n", Pred.dims(0));
				mexPrintf("Pred.dims(1) = %d\n", Pred.dims(1));
				mexPrintf("Pred.dims(2) = %d\n", Pred.dims(2));
				mexEvalString("pause(.0001);");
			}
			Pred = moddims(Pred, Pred.dims(0) * Pred.dims(1), Pred.dims(2));
			if (useKineticModel)
				Pred = join(1, join(0, Pred, constant(0.f, Pred.dims(0), Pred.dims(1))), join(0, constant(0.f, Pred.dims(0), Pred.dims(1)), Pred));
			if (complexType == 3) {
				Pred = join(1, join(0, Pred, constant(0.f, Pred.dims(0), Pred.dims(1))), join(0, constant(0.f, Pred.dims(0), Pred.dims(1)), Pred));
			}
		}
		else {
			const size_t nCol = mxGetN(mxGetField(options, 0, "Pred"));
			const size_t nRow = mxGetM(mxGetField(options, 0, "Pred"));
			Pred = array(nRow, nCol, (float*)mxGetData(mxGetField(options, 0, "Pred")));
		}
	}
	array xlt;
	if (useSmoother) {
		if (complexType == 0)
			xlt = constant(0, imDim, Nt - (window - 1ULL), DimZ);
		else
			xlt = constant(0, imDim, Nt - (window - 1ULL), DimZ, c32);
	}
	if (DEBUG) {
		mexPrintf("useG = %u\n", useG);
		mexEvalString("pause(.0001);");

		mexPrintf("S[0].dims(0)1 = %d\n", S[0].dims(0));
		mexPrintf("S[0].dims(1)1 = %d\n", S[0].dims(1));
	}

	array PP;
	if (storeCovariance == 1)
		if (complexType == 2)
			PP = constant(0.f, Pplus.dims(0), Nt - (window - 1), c32);
		else
			PP = constant(0.f, Pplus.dims(0), Nt - (window - 1));

	DKF(xt, S, Si, m0, Q, Qi, R, Ri, Pplus, Pplusi, Nt, hn, imDim, Nm, storeData, sparseS, complexS, Svalues, Svaluesi, Srow, Scol, sCol, SS3, SS4, complexType, sparseQ, sparseR, sizeQ, sizeR,
		hnU, sizeF, tvF, F, Fi, useF, complexF, useU, useG, sizeG, G, Gi, complexG, u, algorithm, sparseF, fadingMemory, fadingAlpha, covIter, N_lag, useSmoother, approximateKS, steadyS,
		skip, steadyKF, window, regularization, prior, nIter, TV, TVi, Nx, Ny, Nz, Ndx, Ndy, Ndz, gamma, beta, betac, huberDelta, weightsHuber, weightsQuad, LL, Ly,
		lSize, Lvalues, Lcol, LL, Lyi, augType, complexRef, TGV, xlt, useKineticModel, S1, S2, sCols, sRows, RR, NmU, use3D, ensembleSize, computeConsistency, stepSize, initialSteps,
		computeBayesianP, Pred, regType, cgIter, cgThreshold, PP, storeCovariance, computeInitialValue, R2, R2i, Q2, Q2i, forceOrthogonalization, useEnsembleMean);


	af::sync();
	if (complexType > 0) {
		if (DEBUG) {
			mexPrintf("xt.dims(0) = %u\n", xt.dims(0));
			mexPrintf("xt.dims(1) = %u\n", xt.dims(1));
			mexEvalString("pause(.0001);");
		}
		//const mwSize dims[2] = { static_cast<mwSize>(xt.dims(0)), static_cast<mwSize>(xt.dims(1)) };
		//output[0] = mxCreateNumericMatrix(imDim, Nt + 1ULL, mxSINGLE_CLASS, mxCOMPLEX);

		float* apur = real(xt).host<float>();
		float* apui = imag(xt).host<float>();

		float* apuPPr = nullptr;
		float* apuPPi = nullptr;

		if (storeCovariance) {
			apuPPr = real(PP).host<float>();
			std::copy(apuPPr, apuPPr + PP.dims(0) * PP.dims(1) * PP.dims(2) - 1, outP);
			if (complexType == 2) {
				apuPPi = imag(PP).host<float>();
				std::copy(apuPPi, apuPPi + PP.dims(0) * PP.dims(1) * PP.dims(2) - 1, outP2);
			}
		}

		af::sync();

		//std::memcpy(out, apur, (xt.dims(0)* xt.dims(1)) * sizeof(float));
		//std::memcpy(out2, apui, (xt.dims(0)* xt.dims(1)) * sizeof(float));
		std::copy(apur, apur + xt.dims(0) * xt.dims(1) * xt.dims(2) - 1, out);
		std::copy(apui, apui + xt.dims(0) * xt.dims(1) * xt.dims(2) - 1, out2);

		if (useSmoother) {
			float* apurS = real(xlt).host<float>();
			float* apuiS = imag(xlt).host<float>();

			std::memcpy(outS, apurS, (xlt.dims(0) * xlt.dims(1) * xlt.dims(2)) * sizeof(float));
			std::memcpy(out2S, apuiS, (xlt.dims(0) * xlt.dims(1) * xlt.dims(2)) * sizeof(float));
		}

		af::freeHost(apur);
		af::freeHost(apui);
		if (storeCovariance) {
			af::freeHost(apuPPr);
			if (complexType == 2)
				af::freeHost(apuPPi);
		}
	}
	else {
		//output[0] = mxCreateNumericMatrix(xt.dims(0), xt.dims(1), mxSINGLE_CLASS, mxREAL);

		//float* out = (float*)mxGetData(output[0]);
		float* apur = xt.host<float>();
		std::memcpy(out, apur, (xt.dims(0) * xt.dims(1) * xt.dims(2)) * sizeof(float));

		if (useSmoother) {
			float* apurS = real(xlt).host<float>();

			std::memcpy(outS, apurS, (xlt.dims(0) * xlt.dims(1) * xlt.dims(2)) * sizeof(float));
		}
	}


	/**/
}