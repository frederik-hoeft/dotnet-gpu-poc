using ILGPU;
using ILGPU.Runtime;

namespace MLStuff.Numerics;

public sealed class GpuMatrix : GpuProgram<FloatMatrix2D>
{
    private readonly Func<GpuScope, DeviceMatrix> _executeGpu;

    private GpuMatrix(int rows, int columns, Func<GpuScope, DeviceMatrix> executeGpu)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);

        Rows = rows;
        Columns = columns;
        _executeGpu = executeGpu ?? throw new ArgumentNullException(nameof(executeGpu));
    }

    public int Rows { get; }

    public int Columns { get; }

    internal DeviceMatrix ExecuteGpu(GpuScope scope)
    {
        ArgumentNullException.ThrowIfNull(scope);
        return _executeGpu(scope);
    }

    internal override FloatMatrix2D Execute(GpuScope scope)
    {
        DeviceMatrix gpuMatrix = ExecuteGpu(scope);

        scope.Accelerator.Synchronize();

        FloatMatrix2D result = new(gpuMatrix.Rows, gpuMatrix.Columns);
        gpuMatrix.Buffer.CopyToCPU(result.RawData);

        return result;
    }

    public static GpuMatrix From(FloatMatrix2D matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);

        return new GpuMatrix(matrix.Rows, matrix.Columns, scope =>
        {
            DeviceMatrix gpuMatrix = scope.AllocateMatrix(matrix.Rows, matrix.Columns);
            gpuMatrix.Buffer.CopyFromCPU(matrix.RawData);
            return gpuMatrix;
        });
    }

    public static GpuMatrix Multiply(GpuMatrix left, GpuMatrix right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Columns != right.Rows)
        {
            throw new InvalidOperationException(
                $"Cannot multiply {left.Rows}x{left.Columns} by {right.Rows}x{right.Columns}.");
        }

        return new GpuMatrix(left.Rows, right.Columns, scope =>
        {
            DeviceMatrix a = left.ExecuteGpu(scope);
            DeviceMatrix b = right.ExecuteGpu(scope);
            DeviceMatrix result = scope.AllocateMatrix(left.Rows, right.Columns);

            scope.MultiplyMatrixKernel(
                new Index2D(left.Rows, right.Columns),
                a.View,
                b.View,
                result.View,
                left.Columns,
                right.Columns);

            return result;
        });
    }

    public static GpuMatrix Transpose(GpuMatrix source)
    {
        ArgumentNullException.ThrowIfNull(source);

        return new GpuMatrix(source.Columns, source.Rows, scope =>
        {
            DeviceMatrix sourceMatrix = source.ExecuteGpu(scope);
            DeviceMatrix result = scope.AllocateMatrix(source.Columns, source.Rows);

            scope.TransposeKernel(
                new Index2D(source.Rows, source.Columns),
                sourceMatrix.View,
                result.View,
                source.Rows,
                source.Columns);

            return result;
        });
    }

    public static GpuMatrix Apply<TOperation>(GpuMatrix source, TOperation operation)
        where TOperation : unmanaged, IUnaryFloatOperation
    {
        ArgumentNullException.ThrowIfNull(source);

        return new GpuMatrix(source.Rows, source.Columns, scope =>
        {
            DeviceMatrix sourceMatrix = source.ExecuteGpu(scope);
            DeviceMatrix result = scope.AllocateMatrix(source.Rows, source.Columns);

            scope.GetUnaryKernel<TOperation>()(
                new Index1D(source.Rows * source.Columns),
                sourceMatrix.View,
                result.View,
                operation);

            return result;
        });
    }

    public static GpuMatrix Apply<TOperation>(GpuMatrix left, GpuMatrix right, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ValidateSameShape(left, right, nameof(Apply));

        return new GpuMatrix(left.Rows, left.Columns, scope =>
        {
            DeviceMatrix a = left.ExecuteGpu(scope);
            DeviceMatrix b = right.ExecuteGpu(scope);
            DeviceMatrix result = scope.AllocateMatrix(left.Rows, left.Columns);

            scope.GetBinaryKernel<TOperation>()(
                new Index1D(left.Rows * left.Columns),
                a.View,
                b.View,
                result.View,
                operation);

            return result;
        });
    }

    public static GpuMatrix ApplyScalarRight<TOperation>(GpuMatrix left, float right, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation
    {
        ArgumentNullException.ThrowIfNull(left);

        return new GpuMatrix(left.Rows, left.Columns, scope =>
        {
            DeviceMatrix a = left.ExecuteGpu(scope);
            DeviceMatrix result = scope.AllocateMatrix(left.Rows, left.Columns);

            scope.GetScalarRightKernel<TOperation>()(
                new Index1D(left.Rows * left.Columns),
                a.View,
                right,
                result.View,
                operation);

            return result;
        });
    }

    public static GpuMatrix ApplyScalarLeft<TOperation>(float left, GpuMatrix right, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation
    {
        ArgumentNullException.ThrowIfNull(right);

        return new GpuMatrix(right.Rows, right.Columns, scope =>
        {
            DeviceMatrix b = right.ExecuteGpu(scope);
            DeviceMatrix result = scope.AllocateMatrix(right.Rows, right.Columns);

            scope.GetScalarLeftKernel<TOperation>()(
                new Index1D(right.Rows * right.Columns),
                left,
                b.View,
                result.View,
                operation);

            return result;
        });
    }

    public static GpuMatrix Add(GpuMatrix left, GpuMatrix right) =>
        Apply(left, right, new AddFloatOperation());

    public static GpuMatrix Add(GpuMatrix left, float right) =>
        ApplyScalarRight(left, right, new AddFloatOperation());

    public static GpuMatrix Add(float left, GpuMatrix right) =>
        ApplyScalarLeft(left, right, new AddFloatOperation());

    public static GpuMatrix Subtract(GpuMatrix left, GpuMatrix right) =>
        Apply(left, right, new SubtractFloatOperation());

    public static GpuMatrix Subtract(GpuMatrix left, float right) =>
        ApplyScalarRight(left, right, new SubtractFloatOperation());

    public static GpuMatrix Subtract(float left, GpuMatrix right) =>
        ApplyScalarLeft(left, right, new SubtractFloatOperation());

    public static GpuMatrix Max(GpuMatrix left, GpuMatrix right) =>
        Apply(left, right, new MaxFloatOperation());

    public static GpuMatrix Max(GpuMatrix left, float right) =>
        ApplyScalarRight(left, right, new MaxFloatOperation());

    public static GpuMatrix Max(float left, GpuMatrix right) =>
        ApplyScalarLeft(left, right, new MaxFloatOperation());

    public static GpuMatrix Min(GpuMatrix left, GpuMatrix right) =>
        Apply(left, right, new MinFloatOperation());

    public static GpuMatrix Min(GpuMatrix left, float right) =>
        ApplyScalarRight(left, right, new MinFloatOperation());

    public static GpuMatrix Min(float left, GpuMatrix right) =>
        ApplyScalarLeft(left, right, new MinFloatOperation());

    private static void ValidateSameShape(GpuMatrix left, GpuMatrix right, string operationName)
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new InvalidOperationException(
                $"Cannot {operationName} matrices with different shapes: {left.Rows}x{left.Columns} and {right.Rows}x{right.Columns}.");
        }
    }

    public static implicit operator GpuMatrix(FloatMatrix2D matrix) =>
        From(matrix);

    public static GpuMatrix operator *(GpuMatrix left, GpuMatrix right) =>
        Multiply(left, right);

    public static GpuMatrix operator *(GpuMatrix left, FloatMatrix2D right) =>
        Multiply(left, From(right));

    public static GpuMatrix operator *(FloatMatrix2D left, GpuMatrix right) =>
        Multiply(From(left), right);

    public static GpuMatrix operator *(GpuMatrix left, float right) =>
        ApplyScalarRight(left, right, new MultiplyFloatOperation());

    public static GpuMatrix operator *(float left, GpuMatrix right) =>
        ApplyScalarLeft(left, right, new MultiplyFloatOperation());

    public static GpuMatrix operator +(GpuMatrix left, GpuMatrix right) =>
        Add(left, right);

    public static GpuMatrix operator +(GpuMatrix left, FloatMatrix2D right) =>
        Add(left, From(right));

    public static GpuMatrix operator +(FloatMatrix2D left, GpuMatrix right) =>
        Add(From(left), right);

    public static GpuMatrix operator +(GpuMatrix left, float right) =>
        Add(left, right);

    public static GpuMatrix operator +(float left, GpuMatrix right) =>
        Add(left, right);

    public static GpuMatrix operator -(GpuMatrix left, GpuMatrix right) =>
        Subtract(left, right);

    public static GpuMatrix operator -(GpuMatrix left, FloatMatrix2D right) =>
        Subtract(left, From(right));

    public static GpuMatrix operator -(FloatMatrix2D left, GpuMatrix right) =>
        Subtract(From(left), right);

    public static GpuMatrix operator -(GpuMatrix left, float right) =>
        Subtract(left, right);

    public static GpuMatrix operator -(float left, GpuMatrix right) =>
        Subtract(left, right);

    public static GpuMatrix operator -(GpuMatrix source) =>
        source.Negate();
}