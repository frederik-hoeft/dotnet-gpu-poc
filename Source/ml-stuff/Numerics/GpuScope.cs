using ILGPU;
using ILGPU.Runtime;

namespace MLStuff.Numerics;

public sealed class GpuScope(Accelerator accelerator) : IDisposable
{
    private readonly List<IDisposable> _owned = [];
    private readonly Dictionary<Type, object> _unaryKernels = [];
    private readonly Dictionary<Type, object> _binaryKernels = [];
    private readonly Dictionary<Type, object> _scalarRightKernels = [];
    private readonly Dictionary<Type, object> _scalarLeftKernels = [];

    public Accelerator Accelerator { get; } = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

    public TResult Run<TResult>(GpuProgram<TResult> program)
    {
        ArgumentNullException.ThrowIfNull(program);
        return program.Execute(this);
    }

    internal DeviceMatrix AllocateMatrix(int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);

        int length = checked(rows * columns);
        MemoryBuffer1D<float, Stride1D.Dense> buffer = Accelerator.Allocate1D<float>(length);

        _owned.Add(buffer);

        return new DeviceMatrix(rows, columns, buffer);
    }

    internal Action<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> MultiplyMatrixKernel => 
        field ??= Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(GpuKernels.MultiplyMatrix);

    internal Action<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int> TransposeKernel =>
        field ??= Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(GpuKernels.Transpose);

    internal Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation> GetUnaryKernel<TOperation>()
        where TOperation : unmanaged, IUnaryFloatOperation
    {
        Type operationType = typeof(TOperation);

        if (!_unaryKernels.TryGetValue(operationType, out object? boxedKernel))
        {
            boxedKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                TOperation>(GpuKernels.ApplyUnary);

            _unaryKernels.Add(operationType, boxedKernel);
        }

        return (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation>)boxedKernel;
    }

    internal Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation> GetBinaryKernel<TOperation>()
        where TOperation : unmanaged, IBinaryFloatOperation
    {
        Type operationType = typeof(TOperation);

        if (!_binaryKernels.TryGetValue(operationType, out object? boxedKernel))
        {
            boxedKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation>(GpuKernels.ApplyBinary);
            _binaryKernels.Add(operationType, boxedKernel);
        }

        return (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation>)boxedKernel;
    }

    internal Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>, TOperation> GetScalarRightKernel<TOperation>()
        where TOperation : unmanaged, IBinaryFloatOperation
    {
        Type operationType = typeof(TOperation);

        if (!_scalarRightKernels.TryGetValue(operationType, out object? boxedKernel))
        {
            boxedKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>, TOperation>(GpuKernels.ApplyScalarRight);

            _scalarRightKernels.Add(operationType, boxedKernel);
        }

        return (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>, TOperation>)boxedKernel;
    }

    internal Action<Index1D, float, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation> GetScalarLeftKernel<TOperation>()
        where TOperation : unmanaged, IBinaryFloatOperation
    {
        Type operationType = typeof(TOperation);

        if (!_scalarLeftKernels.TryGetValue(operationType, out object? boxedKernel))
        {
            boxedKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, float, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation>(GpuKernels.ApplyScalarLeft);

            _scalarLeftKernels.Add(operationType, boxedKernel);
        }

        return (Action<Index1D, float, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, TOperation>)boxedKernel;
    }

    public void Dispose()
    {
        for (int i = _owned.Count - 1; i >= 0; --i)
        {
            _owned[i].Dispose();
        }

        _owned.Clear();
    }
}
