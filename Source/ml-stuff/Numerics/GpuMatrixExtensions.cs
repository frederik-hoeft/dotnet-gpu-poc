namespace MLStuff.Numerics;

public static class GpuMatrixExtensions
{
    extension(GpuMatrix self)
    {
        public GpuMatrix Multiply(FloatMatrix2D right) => GpuMatrix.Multiply(self, GpuMatrix.From(right));

        public GpuMatrix Multiply(GpuMatrix right) => GpuMatrix.Multiply(self, right);

        public GpuMatrix Multiply(float scalar) => GpuMatrix.ApplyScalarRight(self, scalar, new MultiplyFloatOperation());

        public GpuMatrix Add(FloatMatrix2D right) => GpuMatrix.Add(self, GpuMatrix.From(right));

        public GpuMatrix Add(GpuMatrix right) => GpuMatrix.Add(self, right);

        public GpuMatrix Add(float scalar) => GpuMatrix.Add(self, scalar);

        public GpuMatrix Subtract(FloatMatrix2D right) => GpuMatrix.Subtract(self, GpuMatrix.From(right));

        public GpuMatrix Subtract(GpuMatrix right) => GpuMatrix.Subtract(self, right);

        public GpuMatrix Subtract(float scalar) => GpuMatrix.Subtract(self, scalar);

        public GpuMatrix T => self.Transpose();

        public GpuMatrix Transpose() => GpuMatrix.Transpose(self);

        public GpuMatrix Negate() => GpuMatrix.Apply(self, new NegateFloatOperation());

        public GpuMatrix Abs() => GpuMatrix.Apply(self, new AbsFloatOperation());

        public GpuMatrix Max(FloatMatrix2D right) => GpuMatrix.Max(self, GpuMatrix.From(right));

        public GpuMatrix Max(GpuMatrix right) => GpuMatrix.Max(self, right);

        public GpuMatrix Max(float scalar) => GpuMatrix.Max(self, scalar);

        public GpuMatrix Min(FloatMatrix2D right) => GpuMatrix.Min(self, GpuMatrix.From(right));

        public GpuMatrix Min(GpuMatrix right) => GpuMatrix.Min(self, right);

        public GpuMatrix Min(float scalar) => GpuMatrix.Min(self, scalar);

        public GpuMatrix Apply<TOperation>() where TOperation : unmanaged, IUnaryFloatOperation =>
            GpuMatrix.Apply(self, new TOperation());

        public GpuMatrix Apply<TOperation>(TOperation operation) where TOperation : unmanaged, IUnaryFloatOperation =>
            GpuMatrix.Apply(self, operation);

        public GpuMatrix Apply<TOperation>(FloatMatrix2D right, TOperation operation) where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.Apply(self, GpuMatrix.From(right), operation);

        public GpuMatrix Apply<TOperation>(GpuMatrix right, TOperation operation) where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.Apply(self, right, operation);

        public GpuMatrix ApplyScalarRight<TOperation>(float right, TOperation operation) where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.ApplyScalarRight(self, right, operation);

        public GpuMatrix ApplyScalarLeft<TOperation>(float left, TOperation operation) where TOperation : unmanaged, IBinaryFloatOperation =>
            GpuMatrix.ApplyScalarLeft(left, self, operation);
    }
}