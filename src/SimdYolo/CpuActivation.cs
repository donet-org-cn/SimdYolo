using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdYolo;

/// <summary>提供 FP32 SIMD 指数、Sigmoid 和 SiLU；超出快速指数有效区间时保留框架的特殊值语义。</summary>
internal static class CpuActivation
{
    /// <summary>由环境变量与硬件能力控制的当前 CPU 快路径开关。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_FAST_ACTIVATION") != "0" && Avx2.IsSupported && Fma.IsSupported;
    /// <summary>FP32 快速指数开关，要求 AVX2 和 FMA。</summary>
    private static readonly bool ExpEnabled = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_FAST_EXP") != "0" && Avx2.IsSupported && Fma.IsSupported;
    /// <summary>是否允许 128 位 SiLU 借助 256 位快速指数计算。</summary>
    private static readonly bool Enabled128 = Environment.GetEnvironmentVariable("SIMDYOLO_CPU_SILU128") != "0";
    /// <summary>逐 SIMD 通道计算 SiLU：value/(1+exp(-value))，特殊值沿框架原路径处理。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> Silu(Vector128<float> value)
    {
        if (Enabled128 && Enabled && Sse.MoveMask(Sse.CompareLessThanOrEqual(Sse.And(value, Vector128.Create(0x7fffffff).AsSingle()), Vector128.Create(87f))) == 15)
            return FastSilu(Vector256.Create(value, Vector128<float>.Zero)).GetLower();
        return value / (Vector128<float>.One + Vector128.Exp(-value));
    }
    /// <summary>逐 SIMD 通道计算指数，满足硬件与范围条件时使用 FP32 快速指数。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Exp(Vector256<float> value)
        => ExpEnabled ? FastExp(value) : Vector256.Exp(value);

    /// <summary>逐 SIMD 通道计算指数，满足硬件与范围条件时使用 FP32 快速指数。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> Exp(Vector<float> value)
    {
        if (ExpEnabled && Vector<float>.Count == 8)
        {
            var result = FastExp(Unsafe.As<Vector<float>, Vector256<float>>(ref value));
            return Unsafe.As<Vector256<float>, Vector<float>>(ref result);
        }
        return Vector.Exp(value);
    }

    /// <summary>逐 SIMD 通道计算 Sigmoid：1/(1+exp(-value))。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Sigmoid(Vector256<float> value)
        => Vector256<float>.One / (Vector256<float>.One + Exp(-value));

    /// <summary>逐 SIMD 通道计算 Sigmoid：1/(1+exp(-value))。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> Sigmoid(Vector<float> value)
        => Vector<float>.One / (Vector<float>.One + Exp(-value));

    /// <summary>逐 SIMD 通道计算 SiLU：value/(1+exp(-value))，特殊值沿框架原路径处理。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Silu(Vector256<float> value)
        => Enabled ? FastSilu(value) : value / (Vector256<float>.One + Vector256.Exp(-value));

    /// <summary>逐 SIMD 通道计算 SiLU：value/(1+exp(-value))，特殊值沿框架原路径处理。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> Silu(Vector<float> value)
    {
        if (Enabled && Vector<float>.Count == 8)
        {
            var result = FastSilu(Unsafe.As<Vector<float>, Vector256<float>>(ref value));
            return Unsafe.As<Vector256<float>, Vector<float>>(ref result);
        }
        return value / (Vector<float>.One + Vector.Exp(-value));
    }

    /// <summary>使用快速指数计算八个 FP32 通道的 SiLU。</summary>
    /// <param name="value">待计算的 FP32 值或 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> FastSilu(Vector256<float> value)
    {
        return value / (Vector256<float>.One + FastExp(-value));
    }

    /// <summary>将指数拆成 2 的整数次幂与小区间六阶多项式；任何通道超出 ±87 时整向量回退框架指数。</summary>
    /// <param name="x">待计算指数的 FP32 SIMD 向量。</param>
    /// <returns>按 SIMD 通道分别计算得到的结果向量。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> FastExp(Vector256<float> x)
    {
        var magnitude = Avx.And(x, Vector256.Create(0x7fffffff).AsSingle());
        // 超出快速指数的有效区间时，保留框架对正常值、次正规值、溢出和 NaN 的语义。
        // 不裁剪输入，也不进行重新量化。
        if (Avx.MoveMask(Avx.Compare(magnitude, Vector256.Create(87f), FloatComparisonMode.OrderedLessThanOrEqualNonSignaling)) != 255)
            return Vector256.Exp(x);

        var exponent = Avx.RoundToNearestInteger(x * Vector256.Create(1.4426950408889634f));
        // 将 ln(2) 拆成高位和低位，避免大指数减法丢失区间缩减后的余数。
        // 在 [-ln(2)/2, ln(2)/2] 上使用六阶 Taylor 多项式。
        // 整个多项式仅用 FP32 与 FMA 求值。
        var reduced = Fma.MultiplyAdd(-exponent, Vector256.Create(0.693145751953125f), x);
        reduced = Fma.MultiplyAdd(-exponent, Vector256.Create(1.428606765330187e-6f), reduced);
        var polynomial = Vector256.Create(1f / 720f);
        polynomial = Fma.MultiplyAdd(polynomial, reduced, Vector256.Create(1f / 120f));
        polynomial = Fma.MultiplyAdd(polynomial, reduced, Vector256.Create(1f / 24f));
        polynomial = Fma.MultiplyAdd(polynomial, reduced, Vector256.Create(1f / 6f));
        polynomial = Fma.MultiplyAdd(polynomial, reduced, Vector256.Create(.5f));
        polynomial = Fma.MultiplyAdd(polynomial, reduced, Vector256<float>.One);
        polynomial = Fma.MultiplyAdd(polynomial, reduced, Vector256<float>.One);
        var power = Avx2.ShiftLeftLogical(Avx2.Add(Avx.ConvertToVector256Int32(exponent), Vector256.Create(127)), 23).AsSingle();
        return polynomial * power;
    }
}
