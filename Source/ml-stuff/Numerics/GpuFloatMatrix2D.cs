using ILGPU;
using ILGPU.Runtime;

namespace MLStuff.Numerics;

internal sealed record DeviceMatrix(
    int Rows,
    int Columns,
    MemoryBuffer1D<float, Stride1D.Dense> Buffer)
{
    internal ArrayView1D<float, Stride1D.Dense> View => Buffer.View;
}
