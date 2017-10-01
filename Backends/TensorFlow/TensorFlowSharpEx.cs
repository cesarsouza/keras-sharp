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
        public static TFOutput ReduceMean(this TFGraph g, TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
        {
            if (input.OutputType == TFDataType.Bool)
                input = g.Cast(input, TFDataType.Int8);
            return g.Mean(input, g.ReduceDims(input, axis), keep_dims, operName);
        }


        /// <summary>
        ///   Computes sigmoid cross entropy given `logits`.
        /// </summary>
        /// 
        /// <remarks>
        ///    Measures the probability error in discrete classification tasks in which each
        ///    class is independent and not mutually exclusive.For instance, one could
        ///    perform multilabel classification where a picture can contain both an elephant
        ///    and a dog at the same time.
        /// </remarks>
        /// 
        public static TFOutput sigmoid_cross_entropy_with_logits(this TFGraph g, TFOutput labels, TFOutput logits, string operName = null)
        {
            // https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py#L100

            var scopeName = g.MakeName("logistic_loss", operName);
            using (var newScope = g.WithScope(scopeName))
            {
                //logits = ops.convert_to_tensor(logits, name: "logits");
                //labels = ops.convert_to_tensor(labels, name: "labels");
                //try
                //{
                //    labels.get_shape().merge_with(logits.get_shape())
                //}
                //catch
                //{
                //    throw new ArgumentException("logits and labels must have the same shape ({logits.get_shape()} vs {labels.get_shape()})");
                //}

                // The logistic loss formula from above is
                // x - x * z + log(1 + exp(-x))
                // For x < 0, a more numerically stable formula is
                //   -x * z + log(1 + exp(x))
                // Note that these two expressions can be combined into the following:
                // max(x, 0) - x * z + log(1 + exp(-abs(x)))
                // To allow computing gradients at zero, we define custom versions of max and
                // abs functions.
                TFOutput zeros = g.ZerosLike(logits);
                TFOutput cond = g.GreaterEqual(logits, zeros);
                TFOutput relu_logits = g.Where(cond, logits, zeros);
                TFOutput neg_abs_logits = g.Where(cond, g.Neg(logits), logits);
                return g.Add(
                    g.Sub(relu_logits, g.Mul(logits, labels)),
                    g.Log1p(g.Exp(neg_abs_logits)),
                    operName: operName);
            }
        }

        /// <summary>
        ///   Return elements from x or y depending on condition.
        /// </summary>
        /// 
        /// <param name="condition">LabeledTensor of type `bool`.</param>
        /// <param name="x">LabeledTensor for values where condition is true.</param>
        /// <param name="y">LabeledTensor for values where condition is false.</param>
        /// <param name="name">Optional op name.</param>
        /// 
        /// <returns>The labeled tensor with values according to condition.</returns>
        /// 
        public static TFOutput Where(this TFGraph g, TFOutput condition, TFOutput? x, TFOutput? y, string name= null)
        {
            // https://github.com/tensorflow/tensorflow/blob/d4ce3b4681b3a550c095b2cd18a79494d1cc4039/tensorflow/python/ops/array_ops.py#L2342
            if (x == null && y == null)
                return g.Where(input: condition, operName: name);
            else if (x != null && y != null)
                return g.Select(condition: condition, t: x.Value, e: y.Value, operName: name);
            throw new ArgumentException("x and y must both be non-None or both be None.");
        }

        /*
        public static TFOutput cond(this TFGraph g, TFOutput pred, Func<TFOutput> true_fn = null, Func<TFOutput> false_fn = null, bool strict = false, string operName = null)
        {
            using (var name = g.WithScope(g.MakeName("cond", operName)))
            {
                // Add the Switch to the graph.
                (TFOutput p_2, TFOutput p_1) = g.Switch(pred, pred);
                TFOutput pivot_1 = g.Identity(p_1, operName: "switch_t");
                TFOutput pivot_2 = g.Identity(p_2, operName: "switch_f");
                pred = g.Identity(pred, operName: "pred_id");

                // Disable the fetching of tensors that are only on one branch of cond.
                foreach (TFOutput tensor in new[] { p_1, p_2, pivot_1, pivot_2, pred })
                    g.PreventFetching(tensor.Operation);

                // Build the graph for the true branch in a new context.
                CondContext context_t = new CondContext(pred, pivot_1, branch: 1);
                context_t.Enter();
                (TFTensor orig_res_t, TFTensor res_t) = context_t.BuildCondBranch(true_fn);
                if (orig_res_t == null)
                    throw new ArgumentException("true_fn must have a return value.");
                context_t.ExitResult(res_t);
                context_t.Exit();

                // Build the graph for the false branch in a new context.
                CondContext context_f = new CondContext(pred, pivot_2, branch: 0);
                context_f.Enter();
                (TFTensor orig_res_f, TFTensor res_f) = context_f.BuildCondBranch(false_fn);
                if (orig_res_f == null)
                    throw new ArgumentException("false_fn must have a return value.");
                context_f.ExitResult(res_f);
                context_f.Exit();

                if (!strict)
                {
                    orig_res_t = _UnpackIfSingleton(orig_res_t);
                    orig_res_f = _UnpackIfSingleton(orig_res_f);
                }

                // Check that the return values of the two branches have the same structure.
                try
                {
                    nest.assert_same_structure(orig_res_t, orig_res_f);
                }
                catch (InvalidOperationException e)
                {
                    throw new InvalidOperationException("Incompatible return values of true_fn and false_fn", e);
                }

                // Add the final merge to the graph.
                if (res_t == null)
                    throw new ArgumentException("true_fn and false_fn must return at least one result.");

                TFTensor res_t_flat = nest.flatten(res_t);
                TFTensor res_f_flat = nest.flatten(res_f);

                foreach (var (x, y) in Enumerable.Zip(res_t_flat, res_f_flat, (a, b) => (a, b)))
                {
                    Trace.Assert((isinstance(x, ops.IndexedSlices) &&
                             isinstance(y, ops.IndexedSlices)) ||
                            (isinstance(x, sparse_tensor.SparseTensor) &&
                             isinstance(y, sparse_tensor.SparseTensor)) ||
                            (isinstance(x, ops.Tensor) && isinstance(y, ops.Tensor)));
                    val_x = isinstance(x, ops.Tensor) ? x : x.values;
                    val_y = isinstance(y, ops.Tensor) ? y : y.values;
                    if (val_x.dtype.base_dtype != val_y.dtype.base_dtype)
                        throw new ArgumentException("Outputs of true_fn and false_fn must have the same type: %s, %s" % (val_x.dtype.name, val_y.dtype.name));
                }

                merges = [merge(pair)[0] for pair in zip(res_f_flat, res_t_flat)];
                merges = _convert_flows_to_tensorarrays(nest.flatten(orig_res_t), merges);

                // Add to collections
                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_t);
                ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_f);

                merges = nest.pack_sequence_as(structure: orig_res_t, flat_sequence: merges);

                // Singleton lists and tuples are automatically unpacked if strict == False.
                if (!strict)
                    merges = _UnpackIfSingleton(merges);
                return merges;
            }
        }
        */

        public static void PreventFetching(this TFGraph g, TFOperation op)
        {

        }

        private class CondContext
        {
            private TFOutput pred;
            private TFOutput pivot_1;
            private int branch;

            public CondContext(TFOutput pred, TFOutput pivot_1, int branch)
            {
                this.pred = pred;
                this.pivot_1 = pivot_1;
                this.branch = branch;
            }

            internal (TFTensor, TFTensor) BuildCondBranch(Func<TFOutput> true_fn)
            {
                throw new NotImplementedException();
            }

            internal void Enter()
            {
                throw new NotImplementedException();
            }

            internal void Exit()
            {
                throw new NotImplementedException();
            }

            internal void ExitResult(object res_t)
            {
                throw new NotImplementedException();
            }
        }
        #endregion
    }
}
