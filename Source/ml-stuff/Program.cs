using ILGPU;
using ILGPU.Runtime;
using MLStuff.ML;
using MLStuff.Numerics;

using Context context = Context.CreateDefault();
using Accelerator accelerator = context
    .GetPreferredDevice(preferCPU: false)
    .CreateAccelerator(context);

accelerator.PrintInformation(Console.Out);

using GpuScope gpuScope = new(accelerator);
FloatMatrix2D w = new(
[
    [2, 2],
    [2, 2],
    [-2, -2],
]);
FloatMatrix2D x = new(
[
    [1, 4],
    [2, 5],
    [3, 6],
]);
float b = 0.05f;
GpuMatrix z = (w * x.T) + b;
GpuMatrix a = z.ReLU();
FloatMatrix2D result = gpuScope.Run(a);
Console.WriteLine("x:");
Console.WriteLine(x);
Console.WriteLine("w:");
Console.WriteLine(w);
Console.WriteLine("b:");
Console.WriteLine(b);
Console.WriteLine("z = w * x + b:");
Console.WriteLine(gpuScope.Run(z));
Console.WriteLine("a = ReLU(z):");
Console.WriteLine(result);
