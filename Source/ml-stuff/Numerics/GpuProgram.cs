namespace MLStuff.Numerics;

public abstract class GpuProgram<TResult>
{
    internal abstract TResult Execute(GpuScope scope);
}
