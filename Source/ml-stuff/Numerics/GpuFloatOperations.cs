using ILGPU;

namespace MLStuff.Numerics;

public interface IUnaryFloatOperation
{
    float Apply(float value);
}

public interface IBinaryFloatOperation
{
    float Apply(float left, float right);
}

public readonly struct AddFloatOperation : IBinaryFloatOperation
{
    public float Apply(float left, float right) => left + right;
}

public readonly struct SubtractFloatOperation : IBinaryFloatOperation
{
    public float Apply(float left, float right) => left - right;
}

public readonly struct MultiplyFloatOperation : IBinaryFloatOperation
{
    public float Apply(float left, float right) => left * right;
}

public readonly struct DivideFloatOperation : IBinaryFloatOperation
{
    public float Apply(float left, float right) => left / right;
}

public readonly struct MaxFloatOperation : IBinaryFloatOperation
{
    public float Apply(float left, float right) => IntrinsicMath.Max(left, right);
}

public readonly struct MinFloatOperation : IBinaryFloatOperation
{
    public float Apply(float left, float right) => IntrinsicMath.Min(left, right);
}

public readonly struct AbsFloatOperation : IUnaryFloatOperation
{
    public float Apply(float value) => IntrinsicMath.Abs(value);
}

public readonly struct NegateFloatOperation : IUnaryFloatOperation
{
    public float Apply(float value) => -value;
}
