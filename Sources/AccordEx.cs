using Accord;
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace KerasSharp
{
    public static class AccordEx
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
            Array r = Array.CreateInstance(typeof(T), values.GetLength());

            foreach (int[] idx in r.GetIndices())
                r.SetValue(values.GetValue(idx).To<T>(), idx);

            return r;
        }
        #endregion

    }
}
