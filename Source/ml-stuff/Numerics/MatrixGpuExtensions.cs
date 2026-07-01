namespace MLStuff.Numerics;

public static class MatrixGpuExtensions
{
    extension(FloatMatrix2D self)
    {
        public GpuMatrix ToGpu() => GpuMatrix.From(self);

        public GpuMatrix Multiply(FloatMatrix2D right) =>
            GpuMatrix.From(self).Multiply(right);

        public GpuMatrix Multiply(GpuMatrix right) =>
            GpuMatrix.From(self).Multiply(right);

        public GpuMatrix Add(FloatMatrix2D right) =>
            GpuMatrix.From(self).Add(right);

        public GpuMatrix Add(GpuMatrix right) =>
            GpuMatrix.From(self).Add(right);

        public GpuMatrix Add(float scalar) =>
            GpuMatrix.From(self).Add(scalar);

        public GpuMatrix Subtract(FloatMatrix2D right) =>
            GpuMatrix.From(self).Subtract(right);

        public GpuMatrix Subtract(GpuMatrix right) =>
            GpuMatrix.From(self).Subtract(right);

        public GpuMatrix Subtract(float scalar) =>
            GpuMatrix.From(self).Subtract(scalar);

        public GpuMatrix Max(FloatMatrix2D right) =>
            GpuMatrix.From(self).Max(right);

        public GpuMatrix Max(GpuMatrix right) =>
            GpuMatrix.From(self).Max(right);

        public GpuMatrix Max(float scalar) =>
            GpuMatrix.Max(GpuMatrix.From(self), scalar);

        public GpuMatrix Min(FloatMatrix2D right) =>
            GpuMatrix.From(self).Min(right);

        public GpuMatrix Min(GpuMatrix right) =>
            GpuMatrix.From(self).Min(right);

        public GpuMatrix Min(float scalar) =>
            GpuMatrix.Min(GpuMatrix.From(self), scalar);

        public GpuMatrix Apply<TOperation>(FloatMatrix2D right, TOperation operation)
            where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.From(self).Apply(right, operation);

        public GpuMatrix Apply<TOperation>(GpuMatrix right, TOperation operation)
            where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.From(self).Apply(right, operation);

        public GpuMatrix ApplyScalarRight<TOperation>(float right, TOperation operation)
            where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.From(self).ApplyScalarRight(right, operation);

        public GpuMatrix SubtractFrom(float scalar) =>
            GpuMatrix.Subtract(scalar, GpuMatrix.From(self));

        public GpuMatrix ApplyScalarLeft<TOperation>(float left, TOperation operation)
            where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.From(self).ApplyScalarLeft(left, operation);
        public GpuMatrix Transpose() =>
            GpuMatrix.From(self).Transpose();

        public GpuMatrix Abs() =>
            GpuMatrix.From(self).Abs();

        public GpuMatrix Negate() =>
            GpuMatrix.From(self).Negate();

        public GpuMatrix Apply<TOperation>(TOperation operation) where TOperation : unmanaged, IUnaryFloatOperation =>
            GpuMatrix.From(self).Apply(operation);
    }
}
