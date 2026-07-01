using System;

namespace MLStuff.Numerics;

public sealed class GpuMatrixProgram : GpuProgram<FloatMatrix2D>
{
    private readonly GpuMatrix _inner;

    private GpuMatrixProgram(GpuMatrix inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    public int Rows => _inner.Rows;

    public int Columns => _inner.Columns;

    internal override FloatMatrix2D Execute(GpuScope scope) =>
        _inner.Execute(scope);

    internal DeviceMatrix ExecuteGpu(GpuScope scope) =>
        _inner.ExecuteGpu(scope);

    internal static GpuMatrixProgram Lift(FloatMatrix2D matrix) =>
        new(GpuMatrix.From(matrix));

    internal static GpuMatrixProgram From(GpuMatrix matrix) =>
        new(matrix);

    internal static GpuMatrixProgram Multiply(GpuMatrixProgram left, GpuMatrixProgram right) =>
        From(left._inner.Multiply(right._inner));

    internal static GpuMatrixProgram Apply<TOperation>(GpuMatrixProgram source, TOperation operation)
        where TOperation : unmanaged, IUnaryFloatOperation =>
        From(source._inner.Apply(operation));

    internal static GpuMatrixProgram Apply<TOperation>(GpuMatrixProgram left, GpuMatrixProgram right, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation =>
        From(left._inner.Apply(right._inner, operation));

    internal static GpuMatrixProgram Add(GpuMatrixProgram left, GpuMatrixProgram right) =>
        From(left._inner.Add(right._inner));

    internal static GpuMatrixProgram Add(GpuMatrixProgram left, float right) =>
        From(left._inner.Add(right));

    internal static GpuMatrixProgram Add(float left, GpuMatrixProgram right) =>
        From(GpuMatrix.Add(left, right._inner));

    internal static GpuMatrixProgram Subtract(GpuMatrixProgram left, GpuMatrixProgram right) =>
        From(left._inner.Subtract(right._inner));

    internal static GpuMatrixProgram Subtract(GpuMatrixProgram left, float right) =>
        From(left._inner.Subtract(right));

    internal static GpuMatrixProgram Subtract(float left, GpuMatrixProgram right) =>
        From(GpuMatrix.Subtract(left, right._inner));

    internal static GpuMatrixProgram Max(GpuMatrixProgram left, GpuMatrixProgram right) =>
        From(left._inner.Max(right._inner));

    internal static GpuMatrixProgram Abs(GpuMatrixProgram source) =>
        From(source._inner.Abs());

    internal static GpuMatrixProgram Transpose(GpuMatrixProgram source) =>
        From(source._inner.Transpose());

    public static implicit operator GpuMatrixProgram(FloatMatrix2D matrix) =>
        Lift(matrix);

    public static implicit operator GpuMatrix(GpuMatrixProgram program) =>
        program._inner;

    public static GpuMatrixProgram operator *(GpuMatrixProgram left, GpuMatrixProgram right) =>
        Multiply(left, right);

    public static GpuMatrixProgram operator +(GpuMatrixProgram left, GpuMatrixProgram right) =>
        Add(left, right);

    public static GpuMatrixProgram operator +(GpuMatrixProgram left, float right) =>
        Add(left, right);

    public static GpuMatrixProgram operator +(float left, GpuMatrixProgram right) =>
        Add(left, right);

    public static GpuMatrixProgram operator -(GpuMatrixProgram left, GpuMatrixProgram right) =>
        Subtract(left, right);

    public static GpuMatrixProgram operator -(GpuMatrixProgram left, float right) =>
        Subtract(left, right);

    public static GpuMatrixProgram operator -(float left, GpuMatrixProgram right) =>
        Subtract(left, right);

    public static GpuMatrixProgram operator -(GpuMatrixProgram source) =>
        Apply(source, new NegateFloatOperation());
}
