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
            if (perm == null)
            {
                TFOutput rank = g.Rank(a);
                perm = g.Sub(g.Sub(rank, g.Const(1)), g.Range(g.Const(0), rank, g.Const(1)));
            }

            return g.Transpose(x: a, perm: perm.Value, operName: operName);
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
