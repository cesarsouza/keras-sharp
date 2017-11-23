using KerasSharp.Engine.Topology;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

using static KerasSharp.Python;

namespace KerasSharp
{
    public static class TensorFlowSharpEx
    {


        // Returns range(0, rank(x)) if reduction_indices is null
        public static TFOutput ReduceDims(this TFGraph g, TFOutput input, TFOutput? axis = null)
        {
            if (axis.HasValue)
                return axis.Value;

            // Fast path: avoid creating Rank and Range ops if ndims is known.
            long[] shape = g.GetTensorShape(input).ToArray();
            if (shape.Length >= 0)
            {
                // The python code distinguishes between tensor and sparsetensor

                var array = new int[shape.Length];
                for (int i = 0; i < array.Length; i++)
                    array[i] = i;

                return g.Const(array, TFDataType.Int32);
            }
            return g.Range(g.Const(0), g.Const(shape.Length), g.Const(1));
        }

        #region Staging area - remove after those operations have been implemented in TensorFlowSharp

        public static TFOutput Transpose(this TFGraph g, TFOutput a, TFOutput? perm = null, string operName = null)
        {
            throw new NotImplementedException("https://github.com/migueldeicaza/TensorFlowSharp/pull/178");
        }

        public static TFOutput Cond(this TFGraph g, TFOutput pred, Func<TFOutput> true_fn, Func<TFOutput> false_fn, string operName = null)
        {
            throw new NotImplementedException("https://github.com/migueldeicaza/TensorFlowSharp/pull/176");
        }

        #endregion
    }
}
