# dotnet-gpu-poc

A proof-of-concept exploring monadic patterns in C# for building lazy, composable GPU computation graphs that defer all memory allocation and kernel execution until a single `Run` call, that schedules and executes the entire graph in one shot on the GPU, before returning the result to the CPU.

## Concept

The core idea is that GPU operations are expensive to orchestrate: allocating device memory, copying data, dispatching kernels. Rather than executing eagerly on every operation, this POC models a GPU computation as a **deferred program**: a pure description of *what to compute*, with no GPU side-effects until explicitly run.

`GpuProgram<TResult>` is the monadic type. Each GPU operation (matrix multiply, transpose, add, ReLU, etc.) returns a new `GpuProgram<TResult>` that captures a composed `Func<GpuScope, DeviceMatrix>` - forming an expression tree in memory. Nothing is allocated on the GPU until `GpuScope.Run` is called.

```csharp
// Build a lazy computation graph, no GPU allocations yet
GpuMatrix z = (w * x.T) + b;
GpuMatrix a = z.ReLU();

// Execute the entire graph in one shot, allocating GPU buffers on the way down
FloatMatrix2D result = gpuScope.Run(a);
```

This is structurally similar to the IO monad or `Lazy<T>`: the value describes an effect rather than being one.

## Architecture

| Type | Role |
|---|---|
| `GpuProgram<TResult>` | Abstract monadic base - a deferred GPU computation |
| `GpuMatrix` | Concrete program yielding a `FloatMatrix2D`; wraps `Func<GpuScope, DeviceMatrix>` |
| `GpuScope` | Execution context: owns the ILGPU accelerator, allocates device buffers, caches compiled kernels |
| `FloatMatrix2D` | CPU-side row-major matrix |
| `GpuKernels` | Raw ILGPU kernel implementations (matrix multiply, transpose, elementwise ops) |
| `GpuMatrixExtensions` | C# 14 extension members providing operator-style API (`*`, `+`, `.T`, etc.) |
| `Activations` | ML activation functions built on top of `GpuMatrixExtensions` (`ReLU`, `LeakyReLU`) |

### How composition works

Each operation on a `GpuMatrix` wraps the upstream computation in a new lambda:

```csharp
// GpuMatrix.Multiply (simplified)
return new GpuMatrix(left.Rows, right.Columns, scope =>
{
    DeviceMatrix a = left.ExecuteGpu(scope);   // recurse left
    DeviceMatrix b = right.ExecuteGpu(scope);  // recurse right
    DeviceMatrix result = scope.AllocateMatrix(...);
    scope.MultiplyMatrixKernel(...);
    return result;
});
```

When `scope.Run(program)` is called, the tree is walked depth-first, allocating device memory and dispatching kernels bottom-up.

### Generic kernel dispatch

Element-wise operations use a type-parameterised kernel pattern to avoid code duplication across operations like `Add`, `Subtract`, `Max`, `Negate`, `Abs`:

```csharp
// One kernel handles all unary element-wise operations
public static void ApplyUnary<TOperation>(Index1D index, ..., TOperation operation)
    where TOperation : unmanaged, IUnaryFloatOperation
    => result[index] = operation.Apply(source[index]);
```

`GpuScope` lazily compiles and caches a kernel per `TOperation` type.

## Stack

- **.NET 10** / C# 14
- **[ILGPU](https://ilgpu.net/) 1.5.3** - cross-platform GPU programming for .NET (CUDA, OpenCL, CPU fallback)
- C# 14 **extension members** for ergonomic operator syntax on `GpuMatrix`

## Running

```sh
cd Source
dotnet run --project ml-stuff
```

The program will auto-select a GPU if available, falling back to the ILGPU CPU accelerator otherwise. It prints the result of a small forward-pass computation: $a = \mathrm{ReLU}(W \cdot X^T + b)$.
