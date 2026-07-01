using ILGPU;
using ILGPU.Runtime;

namespace MLStuff.Numerics;

public static class GpuKernels
{
    public static void MultiplyMatrix(Index2D index, ArrayView1D<float, Stride1D.Dense> left, ArrayView1D<float, Stride1D.Dense> right, ArrayView1D<float, Stride1D.Dense> result, int leftColumns, int rightColumns)
    {
        int row = index.X;
        int column = index.Y;

        float sum = 0f;

        for (int k = 0; k < leftColumns; ++k)
        {
            float a = left[(row * leftColumns) + k];
            float b = right[(k * rightColumns) + column];

            sum += a * b;
        }

        result[(row * rightColumns) + column] = sum;
    }

    public static void Transpose(Index2D index, ArrayView1D<float, Stride1D.Dense> source, ArrayView1D<float, Stride1D.Dense> result, int sourceRows, int sourceColumns)
    {
        int sourceRow = index.X;
        int sourceColumn = index.Y;
        result[(sourceColumn * sourceRows) + sourceRow] = source[(sourceRow * sourceColumns) + sourceColumn];
    }

    public static void ApplyUnary<TOperation>(Index1D index, ArrayView1D<float, Stride1D.Dense> source, ArrayView1D<float, Stride1D.Dense> result, TOperation operation)
        where TOperation : unmanaged, IUnaryFloatOperation
        => result[index] = operation.Apply(source[index]);

    public static void ApplyBinary<TOperation>(Index1D index, ArrayView1D<float, Stride1D.Dense> left, ArrayView1D<float, Stride1D.Dense> right, ArrayView1D<float, Stride1D.Dense> result, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation
        => result[index] = operation.Apply(left[index], right[index]);

    public static void ApplyScalarRight<TOperation>(Index1D index, ArrayView1D<float, Stride1D.Dense> left, float right, ArrayView1D<float, Stride1D.Dense> result, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation
        => result[index] = operation.Apply(left[index], right);

    public static void ApplyScalarLeft<TOperation>(Index1D index, float left, ArrayView1D<float, Stride1D.Dense> right, ArrayView1D<float, Stride1D.Dense> result, TOperation operation)
        where TOperation : unmanaged, IBinaryFloatOperation
        => result[index] = operation.Apply(left, right[index]);
}
