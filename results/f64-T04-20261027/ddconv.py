"""f64-T04 step 2 helper: the double-float (df64) source transformation of the streaming forward kernel.

The vendored kernel (fastmace/kernels/oeq_streaming/factorized_projected.cuh, rendered for float64) is cut down to
its preamble and ``oeq_projected_forward``; its arithmetic type ``scalar_t`` becomes a C++ struct ``df`` (a pair of
float32, hi + lo) with overloaded ``+``, ``*`` and ``+=``, while the kernel's pointer arguments stay ``double*``.
Every value loaded from float64 memory converts to a pair; every product and sum runs in double-float on the FP32
pipe; the output converts back to float64 at the store. The kernel's index arithmetic and control flow are
unchanged.

Arithmetic: Joldes, Muller, Popescu, "Tight and rigorous error bounds for basic building blocks of double-word
arithmetic", ACM TOMS 44(2), 2017 - DWTimesDW3 (Alg. 12, FMA-based, relative error <= 4u^2) for every
product (coefficients included), and AccurateDWPlusDW (Alg. 6, <= 3u^2 + 13u^3) for df+df, u = 2^-24.
Two-sum and two-prod are written with __fadd_rn / __fmul_rn / __fmaf_rn and the program is compiled with
--fmad=false, so the compiler can neither contract nor reorder them.

float64 -> pair, two variants:
  cvt   hi = __double2float_rn(x), lo = __double2float_rn(x - hi)      (F2F conversions and a DADD)
  bits  hi = x's top 24 significand bits (truncated) by integer ops, lo = the remaining 29 bits through
        I2F and two FP32 multiplies, then Fast2Sum(hi, lo) to normalise; |x| < 2^-126 flushes to 0
A float64 has 53 significand bits and a pair carries at most 49, so the conversion itself has relative error up
to about 2^-49 (1.8e-15) in both variants.
Coefficients (the kernel's float64 literals) become pairs at compile time (constant folded).
"""
import re

DF_HEADER = r"""
struct df { float hi, lo; };

__device__ __forceinline__ df dd_two_sum(float a, float b) {
    float s = __fadd_rn(a, b); float bb = __fadd_rn(s, -a);
    return {s, __fadd_rn(__fadd_rn(a, -__fadd_rn(s, -bb)), __fadd_rn(b, -bb))};
}
__device__ __forceinline__ df dd_fast_two_sum(float a, float b) {
    float s = __fadd_rn(a, b); float z = __fadd_rn(s, -a);
    return {s, __fadd_rn(b, -z)};
}
__device__ __forceinline__ df operator*(df x, df y) {          // DWTimesDW3, Alg. 12
    float ch = __fmul_rn(x.hi, y.hi);
    float cl1 = __fmaf_rn(x.hi, y.hi, -ch);
    float tl0 = __fmul_rn(x.lo, y.lo);
    float tl1 = __fmaf_rn(x.hi, y.lo, tl0);
    float cl2 = __fmaf_rn(x.lo, y.hi, tl1);
    return dd_fast_two_sum(ch, __fadd_rn(cl1, cl2));
}
__device__ __forceinline__ df operator+(df x, df y) {          // AccurateDWPlusDW, Alg. 6
    df s = dd_two_sum(x.hi, y.hi); df t = dd_two_sum(x.lo, y.lo);
    df v = dd_fast_two_sum(s.hi, __fadd_rn(s.lo, t.hi));
    return dd_fast_two_sum(v.hi, __fadd_rn(t.lo, v.lo));
}
__device__ __forceinline__ df& operator+=(df& x, df y) { x = x + y; return x; }

// Compile-time constant -> pair (the kernel's literals); never used on loaded data.
__device__ __forceinline__ df const_df(double c) { float h = (float)c; return {h, (float)(c - (double)h)}; }
__device__ __forceinline__ df operator*(df x, double c) { return x * const_df(c); }

#if DF_CONVERT_BITS
__device__ __forceinline__ df load_df(double x) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(x);
    const int e = int((b >> 52) & 0x7ff) - 1023;               // unbiased exponent
    if (e < -126) return {0.0f, 0.0f};                          // tiny (or zero/subnormal): flush
    const unsigned int sign = (unsigned int)(b >> 63) << 31;
    const float hi = __int_as_float(sign | (unsigned int)(e + 127) << 23 | (unsigned int)((b >> 29) & 0x7fffff));
    const unsigned int r = (unsigned int)(b & 0x1fffffffull);   // low 29 bits, weight 2^(e-52)
    float lo = __fmul_rn(__fmul_rn(__uint2float_rn(r), 2.2204460492503131e-16f),   // 2^-52
                         __int_as_float((unsigned int)(e + 127) << 23));          // 2^e
    lo = sign ? -lo : lo;
    return dd_fast_two_sum(hi, lo);
}
#else
__device__ __forceinline__ df load_df(double x) {
    const float h = __double2float_rn(x);
    return {h, __double2float_rn(x - (double)h)};
}
#endif
__device__ __forceinline__ double store_df(df x) { return (double)x.hi + (double)x.lo; }
"""


def forward_only(source):
    """The preamble and oeq_projected_forward of the rendered streaming source (other kernels dropped)."""
    start = source.index('extern "C" __global__ void oeq_projected_forward(')
    stop = source.index('extern "C" __global__', start + 10)
    preamble = source[:start]
    # Drop preamble helpers that use floating-point atomics (only the reverse kernels use them).
    return preamble, source[start:stop]


def dd_forward_source(source, convert="cvt"):
    """Transform the rendered float64 streaming source into the df64 forward kernel source."""
    if "using scalar_t = double;" not in source:
        raise ValueError("expected a float64 rendering of the streaming kernels")
    preamble, kernel = forward_only(source)
    preamble = preamble.replace("using scalar_t = double;", "using scalar_t = df;")
    preamble = preamble.replace("using acc_t = double;", "using acc_t = df;")
    preamble = (f"#define DF_CONVERT_BITS {1 if convert == 'bits' else 0}\n" + DF_HEADER
                + preamble)
    kernel = kernel.replace("const scalar_t* ", "const double* ").replace("scalar_t* out", "double* out")
    # Loads: `x[...]`, `sh[...]`, `weights[...]` -> load_df(...).
    kernel = re.sub(r"\b(x|sh|weights)\[([^\]]+)\]", r"load_df(\1[\2])", kernel)
    # Store.
    kernel, count = re.subn(r"out\[output_index\] = (\w+);", r"out[output_index] = store_df(\1);", kernel)
    if not count:
        raise ValueError("store pattern not found")
    # Coefficient literals (hex float64) -> pairs folded at compile time.
    kernel = kernel.replace("static_cast<scalar_t>(", "const_df(")
    # The kernel's literal zero initialisers.
    kernel = kernel.replace("scalar_t(0)", "df{0.0f, 0.0f}")
    # Rename so both kernels can live side by side.
    kernel = kernel.replace("oeq_projected_forward(", "oeq_projected_forward_dd(", 1)
    return preamble + kernel
