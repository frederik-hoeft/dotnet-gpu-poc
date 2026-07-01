using System;
using System.Text;

namespace MLStuff.Numerics;

public sealed class FloatMatrix2D
{
    public FloatMatrix2D(int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);

        Rows = rows;
        Columns = columns;
        RawData = new float[checked(rows * columns)];
    }

    public FloatMatrix2D(float[][] data)
        : this(ValidateRows(data), ValidateColumns(data))
    {
        for (int i = 0; i < data.Length; ++i)
        {
            if (data[i].Length != Columns)
            {
                throw new ArgumentException("All rows must have the same number of columns.", nameof(data));
            }

            data[i].CopyTo(RawData, i * Columns);
        }
    }

    public FloatMatrix2D(int rows, int columns, ReadOnlySpan<float> data)
        : this(rows, columns)
    {
        if (data.Length != RawData.Length)
        {
            throw new ArgumentException("Input data length does not match matrix dimensions.", nameof(data));
        }

        data.CopyTo(RawData);
    }

    public ReadOnlySpan<float> Data => RawData;

    internal float[] RawData { get; }

    public int Rows { get; }

    public int Columns { get; }

    public float this[int row, int column]
    {
        get => RawData[GetFlatIndex(row, column)];
        set => RawData[GetFlatIndex(row, column)] = value;
    }

    public override string ToString()
    {
        StringBuilder sb = new();
        sb.AppendLine($"FloatMatrix2D ({Rows}x{Columns}):");

        for (int i = 0; i < Rows; ++i)
        {
            sb.Append("  [");

            for (int j = 0; j < Columns; ++j)
            {
                sb.Append(this[i, j].ToString("G4"));

                if (j < Columns - 1)
                {
                    sb.Append(", ");
                }
            }

            sb.AppendLine("]");
        }

        return sb.ToString();
    }

    private int GetFlatIndex(int row, int column)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Rows);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, Columns);

        return (row * Columns) + column;
    }

    private static int ValidateRows(float[][] data)
    {
        ArgumentNullException.ThrowIfNull(data);

        if (data.Length == 0)
        {
            throw new ArgumentException("Matrix must have at least one row.", nameof(data));
        }

        return data.Length;
    }

    private static int ValidateColumns(float[][] data)
    {
        ArgumentNullException.ThrowIfNull(data);

        if (data.Length == 0)
        {
            throw new ArgumentException("Matrix must have at least one row.", nameof(data));
        }

        if (data[0] is null || data[0].Length == 0)
        {
            throw new ArgumentException("Matrix must have at least one column.", nameof(data));
        }

        return data[0].Length;
    }

    public GpuMatrix T => GpuMatrix.From(this).Transpose();

    public static GpuMatrix operator *(FloatMatrix2D left, FloatMatrix2D right) =>
        GpuMatrix.From(left).Multiply(right);

    public static GpuMatrix operator +(FloatMatrix2D left, FloatMatrix2D right) =>
        GpuMatrix.From(left).Add(right);

    public static GpuMatrix operator +(FloatMatrix2D left, float right) =>
        GpuMatrix.From(left).Add(right);

    public static GpuMatrix operator +(float left, FloatMatrix2D right) =>
        GpuMatrix.Add(left, GpuMatrix.From(right));

    public static GpuMatrix operator -(FloatMatrix2D left, FloatMatrix2D right) =>
        GpuMatrix.From(left).Subtract(right);

    public static GpuMatrix operator -(FloatMatrix2D left, float right) =>
        GpuMatrix.From(left).Subtract(right);

    public static GpuMatrix operator -(float left, FloatMatrix2D right) =>
        GpuMatrix.Subtract(left, GpuMatrix.From(right));

    public static GpuMatrix operator -(FloatMatrix2D source) =>
        GpuMatrix.From(source).Negate();
}
