using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

using static KerasSharp.Python;

namespace KerasSharp
{
    public static class TensorFlowSharpEx
    {
        static string MakeName(this TFGraph g, string operName, string userName)
        {
            if (userName == null)
            {
                var k = g.CurrentNameScope == "" ? operName : g.CurrentNameScope + "/" + operName;

                return str(id(k));
            }
            if (g.CurrentNameScope == "")
                return userName;
            return g.CurrentNameScope + "/" + userName;
        }

        // Returns range(0, rank(x)) if reduction_indices is null
        static TFOutput ReduceDims(this TFGraph g, TFOutput input, TFOutput? axis = null)
        {
            if (axis.HasValue)
                return axis.Value;

            // Fast path: avoid creating Rank and Range ops if ndims is known.
            var shape = g.GetTensorShape(input);
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

        /// <summary>
        /// Clips tensor values to a specified min and max.
        /// </summary>
        /// <remarks>
        /// Given a tensor <paramref name="x"/>, this operation returns a tensor of the same type and shape
        /// as <paramref name="x"/> with its values clipped to <paramref name="clip_value_min"/> and <paramref name="clip_value_max"/>.
        /// Any values less than <paramref name="clib_value_min"/> are set to <paramref name="clip_value_min"/>. Any values greater than 
        /// <paramref name="clip_value_max"/> are set to <paramref name="clip_value_max"/>.
        /// </remarks>
        /// <param name="x">The tensor.</param>
        /// <param name="clip_value_min">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
        /// <param name="clip_value_max">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
        /// <param name="operName">Operation name, optional.</param>
        /// <returns>A clipped <see cref="TFOutput">tensor</see>.</returns>
        public static TFOutput ClipByValue(this TFGraph g, TFOutput x, TFOutput clip_value_min, TFOutput clip_value_max, string operName = null)
        {
            // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L33
            var scopeName = g.MakeName("ClipByValue", operName);
            using (var newScope = g.WithScope(scopeName))
            {
                // Go through list of tensors, for each value in each tensor clip
                var t_min = g.Minimum(x, clip_value_max);
                var t_max = g.Maximum(t_min, clip_value_min, operName: operName);
                return t_max;
            }
        }

        /// <summary>
        /// Computes the mean of elements across dimensions of a tensor.
        /// </summary>
        /// <returns>The reduced tensor.</returns>
        /// <param name="input">The tensor to reduce. Should have numeric type.</param>
        /// <param name="axis">The dimensions to reduce. If not set (the default), reduces all dimensions.</param>
        /// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
        /// <param name="operName">A name for the operation, optional.</param>
        /// <remarks>
        /// <para>
        ///   Reduces input_tensor along the dimensions given in axis.
        /// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
        /// entry in axis. If keep_dims is true, the reduced dimensions
        /// are retained with length 1.</para>
        /// 
        /// <para>
        /// If axis has no entries, all dimensions are reduced, and a
        /// tensor with a single element is returned.</para>
        /// </remarks>
        public static TFOutput ReduceMean(this  TFGraph g, TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
        {
            return g.Mean(input, g.ReduceDims(input, axis), keep_dims, operName);
        }

        #endregion
    }
}
