using Accord;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Accord.Math
{
    public static class MatrixEx
    {


        #region Staging area - remove after those operations have been implemented in Accord.NET

        public static Array ExpandDimensions(this Array array, int axis)
        {
            List<int> dimensions = array.GetLength().ToList();
            dimensions.Insert(axis, 1);
            Array res = Array.CreateInstance(array.GetInnerMostType(), dimensions.ToArray());
            Buffer.BlockCopy(array, 0, res, 0, res.GetNumberOfBytes());
            return res;
        }

        public static Array Squeeze(this Array array)
        {
            int[] dimensions = array.GetLength().Where(x => x != 1).ToArray();
            Array res = Array.CreateInstance(array.GetInnerMostType(), dimensions);
            Buffer.BlockCopy(array, 0, res, 0, res.GetNumberOfBytes());
            return res;
        }

        public static Array Flatten(this Array array, MatrixOrder order = MatrixOrder.CRowMajor)
        {
            Type t = array.GetInnerMostType();

            if (order == MatrixOrder.CRowMajor)
            {
                Array dst = Array.CreateInstance(t, array.Length);
                Buffer.BlockCopy(array, 0, dst, 0, dst.Length * Marshal.SizeOf(t));
                return dst;
            }
            else
            {
                Array r = Array.CreateInstance(t, array.Length);

                int c = 0;
                foreach (int[] idx in array.GetIndices())
                    r.SetValue(value: array.GetValue(idx.Reversed()), index: c++);

                return r;
            }
        }


        public static int GetNumberOfBytes(this Array array)
        {
            Type elementType = array.GetInnerMostType();
            int elementSize = Marshal.SizeOf(elementType);
            int numberOfElements = array.GetTotalLength();
            return elementSize * numberOfElements;
        }

        public static double?[] ones(int length)
        {
            double?[] r = new double?[length];
            for (int i = 0; i < r.Length; i++)
                r[i] = 1;
            return r;
        }

        public static Array Transpose(this Array array)
        {
            Array result = Array.CreateInstance(array.GetInnerMostType(), array.GetLength().Reversed());
            foreach (var idx in array.GetIndices())
            {
                object v = array.GetValue(idx);
                result.SetValue(v, idx.Reversed());
            }

            return result;
        }

        public static T[][] ToJagged<T>(this IList<IList<T>> values)
        {
            return values.Apply(x => x.ToArray());
        }

        public static int Rows<T>(this IList<IList<T>> values)
        {
            return values.Count;
        }

        public static int Columns<T>(this IList<IList<T>> values)
        {
            return values[0].Count;
        }

        public static T[,] ToMatrix<T>(this IList<IList<T>> values)
        {
            int rows = values.Rows();
            int cols = values.Columns();

            T[,] result = Matrix.Zeros<T>(rows, cols);
            for (int i = 0; i < values.Count; i++)
                for (int j = 0; j < values[i].Count; j++)
                    result[i, j] = values[i][j];

            return result;
        }

        public static Array ToMatrix<T>(this IList<T> values, int[] shape, MatrixOrder order = MatrixOrder.CRowMajor)
        {
            if (order != MatrixOrder.CRowMajor)
            {
                Array r = Array.CreateInstance(typeof(T), shape.Reversed());

                int c = 0;
                foreach (int[] idx in r.GetIndices())
                    r.SetValue(value: values[c++], indices: idx);

                return r.Transpose();
            }
            else
            {
                Array r = Array.CreateInstance(typeof(T), shape);

                int c = 0;
                foreach (int[] idx in r.GetIndices())
                    r.SetValue(value: values[c++], indices: idx);

                return r;
            }
        }

        public static Array Convert<T>(this Array values)
        {
            return Convert(values, typeof(T));
        }

        public static Array Convert(this Array values, Type type)
        {
            Array r = Array.CreateInstance(type, values.GetLength());

            foreach (int[] idx in r.GetIndices())
                r.SetValue(To(values.GetValue(idx), type), idx);

            return r;
        }

        /// <summary>
        ///   Converts an object into another type, irrespective of whether
        ///   the conversion can be done at compile time or not. This can be
        ///   used to convert generic types to numeric types during runtime.
        /// </summary>
        /// 
        /// <param name="value">The value to be converted.</param>
        /// 
        /// <returns>The result of the conversion.</returns>
        /// 
        public static T To<T>(this object value)
        {
            return (T)To(value, typeof(T));
        }

        /// <summary>
        ///   Converts an object into another type, irrespective of whether
        ///   the conversion can be done at compile time or not. This can be
        ///   used to convert generic types to numeric types during runtime.
        /// </summary>
        /// 
        /// <param name="value">The value to be converted.</param>
        /// 
        /// <returns>The result of the conversion.</returns>
        /// 
        public static object To(this object value, Type type)
        {
            if (value == null)
                return System.Convert.ChangeType(null, type);

            if (value is IConvertible)
                return System.Convert.ChangeType(value, type);

            MethodInfo[] methods = type.GetMethods(BindingFlags.Public | BindingFlags.Static);

            foreach (MethodInfo m in methods)
            {
                if (m.IsPublic && m.IsStatic)
                {
                    if ((m.Name == "op_Implicit" || m.Name == "op_Explicit") && m.ReturnType == type)
                        return m.Invoke(null, new[] { value });
                }
            }

            if (value is Array && type.IsArray)
            {
                Array v = value as Array;
                int rank = type.GetArrayRank();
                int[] length = v.GetLength();
                int[] expected = length.Get(rank, 0);
                if (expected.IsEqual(1))
                {
                    int[] first = length.Get(0, rank);
                    var elementType = type.GetInnerMostType();

                    var dst = Array.CreateInstance(elementType, first);
                    Buffer.BlockCopy(v, 0, dst, 0, v.Length * Marshal.SizeOf(elementType));
                    return dst;
                }
            }

            return System.Convert.ChangeType(value, type);
        }

        public static Type GetInnerMostType(this Type type)
        {
            while (type.IsArray)
                type = type.GetElementType();
            return type;
        }


        public static Array Get(Array array, int dimension, int[] indices)
        {
            int[] lengths = array.GetLength();
            lengths[dimension] = indices.Length;

            Type type = array.GetInnerMostType();
            Array r = Array.CreateInstance(type, lengths);

            for (int i = 0; i < indices.Length; i++)
                Set(r, dimension: 0, index: i, value: Get(array, dimension: 0, index: i));

            return r;
        }

        public static Array Get(Array array, int dimension, int index)
        {
            return Get(array, dimension, index, index + 1);
        }

        public static Array Get(Array array, int dimension, int start, int end)
        {
            int[] length = array.GetLength();
            length = length.RemoveAt(dimension);
            int rows = end - start;
            if (length.Length == 0)
                length = new int[] { rows };

            Type type = array.GetInnerMostType();
            Array r = Array.CreateInstance(type, length);
            int rowSize = array.Length / array.GetLength(dimension);
            Buffer.BlockCopy(array, start * rowSize * Marshal.SizeOf(type), r, 0, rows * rowSize * Marshal.SizeOf(type));
            return r;
        }

        public static void Set(Array array, int dimension, int index, Array value)
        {
            Set(array, dimension, index, index + 1, value);
        }

        public static void Set(Array array, int dimension, int start, int end, Array value)
        {
            Type type = array.GetInnerMostType();
            int rowSize = array.Length / array.GetLength(0);
            int length = end - start;
            Buffer.BlockCopy(value, 0, array, start * rowSize * Marshal.SizeOf(type), length * rowSize * Marshal.SizeOf(type));
        }

        public static bool IsSquare(this Array array)
        {
            int first = array.GetLength(0);
            for (int i = 1; i < array.Rank; i++)
                if (array.GetLength(i) != first)
                    return false;
            return true;
        }

        public static bool IsLessThan<T, U>(this T[] a, U[] b)
            where T : IComparable
        {
            for (int i = 0; i < a.Length; i++)
                if (a[i].IsLessThan(b[i]))
                    return false;
            return true;
        }

        public static bool IsLessThanOrEqual<T, U>(this T[] a, U[] b)
            where T : IComparable
        {
            for (int i = 0; i < a.Length; i++)
                if (a[i].IsLessThanOrEqual(b[i]))
                    return false;
            return true;
        }

        public static bool IsGreaterThan<T, U>(this T[] a, U[] b)
            where T : IComparable
        {
            for (int i = 0; i < a.Length; i++)
                if (a[i].IsGreaterThan(b[i]))
                    return false;
            return true;
        }

        public static bool IsGreaterThanOrEqual<T, U>(this T[] a, U[] b)
            where T : IComparable
        {
            for (int i = 0; i < a.Length; i++)
                if (a[i].IsGreaterThanOrEqual(b[i]))
                    return false;
            return true;
        }

        public static Array Zeros(Type type, int[] shape)
        {
            return Array.CreateInstance(type, shape);
        }

        public static Array Create<T>(int[] shape, T value)
        {
            return Create(typeof(T), shape, value);
        }

        public static Array Create(Type type, int[] shape, object value)
        {
            Array arr = Array.CreateInstance(type, shape);
            foreach (int[] idx in arr.GetIndices())
                arr.SetValue(value, idx);
            return arr;
        }


        #endregion

    }

}
