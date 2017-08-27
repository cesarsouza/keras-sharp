using Accord.Math;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

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
        #endregion

    }
}
