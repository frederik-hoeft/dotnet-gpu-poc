using MLStuff.Numerics;

namespace MLStuff.ML;

public static class Activations
{
    extension(GpuMatrix z)
    {
        public GpuMatrix ReLU() => z.Max(0);

        public GpuMatrix LeakyReLU(float alpha) => z.Max(z * alpha);
    }
}