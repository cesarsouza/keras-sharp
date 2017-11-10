// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

namespace KerasSharp.Models
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using KerasSharp.Engine;
    using System.Diagnostics;
    using KerasSharp.Metrics;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Losses;

    using static KerasSharp.Backends.Current;
    using static KerasSharp.Python;

    using Accord.Math;
    using System.Collections;
    using KerasSharp.Optimizers;


    public enum Shuffle
    {
        True,
        False,
        Batch // This looks a lot like https://thedailywtf.com/articles/What_Is_Truth_0x3f_...
    }

    public static class ExtensionMethods
    {
        public static IEnumerable<T> Concatenate<T>(this IEnumerable<T>[] lists)
        {
            return lists.SelectMany(x => x);
        }
    }


    public abstract class Function
    {
        public Function()
        { 
        }

        public abstract List<Tensor> Call(List<Array> ins);
    }

    /// <summary>
    ///   The Model class adds training & evaluation routines to a <see cref="Container"/>.
    /// </summary>
    /// 
    public partial class Model : Container
    {

        protected internal virtual Sequential callback_model { get; set; }
        public IOptimizer optimizer;
        public Dictionary<string, string> sample_weight_mode;
        public Dictionary<string, ILoss> loss;
        public Dictionary<string, double> loss_weights;
        public Tensor total_loss;
        public List<Tensor> sample_weights;
        protected List<ILoss> loss_functions;
        protected List<Tensor> _feed_outputs;
        protected List<string> _feed_output_names;
        protected List<int?[]> _feed_output_shapes;
        protected List<ILoss> _feed_loss_fns;
        public List<Tensor> targets;
        protected List<Tensor> _feed_targets;
        public Dictionary<string, List<IMetric>> metrics;
        public List<string> metrics_names;
        public List<Tensor> metrics_tensors;
        protected List<Tensor> _feed_sample_weights;
        protected List<Tensor> _collected_trainable_weights;
        protected Function train_function;
        protected Function test_function;
        protected Function predict_function;
        protected bool stop_training;
        protected History history;
        private List<string> sample_weight_modes;
        public List<string> _feed_sample_weight_modes;



        public Model()
        {
        }

        public Model(List<Tensor> inputs, List<Tensor> outputs, string name = null)
            : base(inputs, outputs, name)
        {
        }


        public void Compile(IOptimizer optimizer, ILoss loss, params IMetric[] metrics)
        {
            Compile(optimizer, loss.dict_from_single(), new List<IMetric>(metrics).dict_from_single());
        }

        /// <summary>
        ///   Configures the model for training.
        /// </summary>
        /// 
        /// <param name="optimizer">The optimization algorithm.</param>
        /// <param name="loss">The objective function (to be minimized). model has multiple outputs, you can use a different loss
        ///   on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model 
        ///   will then be the sum of all individual losses.</param>
        /// <param name="metrics">The list of metrics to be evaluated by the model during training and testing. Typically you 
        ///   will use `metrics =['accuracy']`. To specify different metrics for different outputs of a multi - output model, 
        ///   you could also pass a dictionary, such as `metrics ={ 'output_a': 'accuracy'}`.</param>
        /// <param name="loss_weights">The optional list or dictionary specifying scalar coefficients (Python floats) to weight 
        ///   the loss contributions of different model outputs. The loss value that will be minimized by the model will then be 
        ///     the *weighted sum* of all individual losses, weighted by the `loss_weights` coefficients. If a list, it is expected 
        ///     to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names(strings) to scalar 
        ///     coefficients.</param>
        /// <param name="sample_weight_mode">If you need to do timestep - wise sample weighting(2D weights), set this to `"temporal"`. 
        ///   `null` defaults to sample - wise weights(1D). If the model has multiple outputs, you can use a different `sample_weight_mode` 
        ///     on each output by passing a dictionary or a list of modes.</param>
        /// 
        public void Compile(IOptimizer optimizer, ILoss loss, List<IMetric> metrics = null)
        {
            Compile(optimizer, loss.dict_from_single(), metrics.dict_from_single());
        }

        public void Compile(IOptimizer optimizer, List<ILoss> loss, List<List<IMetric>> metrics = null)
        {
            Compile(optimizer, loss.dict_from_list(), metrics.dict_from_list());
        }

        public void Compile(string optimizer, string loss, string[] metrics = null)
        {
            Compile(Get<IOptimizer>(optimizer), Get<ILoss>(loss), Get<IMetric>(metrics));
        }

        public void Compile(IOptimizer optimizer, string loss, string[] metrics = null)
        {
            Compile(optimizer, Get<ILoss>(loss), Get<IMetric>(metrics));
        }

        public void Compile(string optimizer, string loss, object[] metrics)
        {
            Compile(Get<IOptimizer>(optimizer), Get<ILoss>(loss), Get<IMetric>(metrics));
        }

        /// <summary>
        ///   Configures the model for training.
        /// </summary>
        /// 
        /// <param name="optimizer">The optimization algorithm.</param>
        /// <param name="loss">The objective function (to be minimized). model has multiple outputs, you can use a different loss
        ///   on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model 
        ///   will then be the sum of all individual losses.</param>
        /// <param name="metrics">The list of metrics to be evaluated by the model during training and testing. Typically you 
        ///   will use `metrics =['accuracy']`. To specify different metrics for different outputs of a multi - output model, 
        ///   you could also pass a dictionary, such as `metrics ={ 'output_a': 'accuracy'}`.</param>
        /// <param name="loss_weights">The optional list or dictionary specifying scalar coefficients (Python floats) to weight 
        ///   the loss contributions of different model outputs. The loss value that will be minimized by the model will then be 
        ///     the *weighted sum* of all individual losses, weighted by the `loss_weights` coefficients. If a list, it is expected 
        ///     to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names(strings) to scalar 
        ///     coefficients.</param>
        /// <param name="sample_weight_mode">If you need to do timestep - wise sample weighting(2D weights), set this to `"temporal"`. 
        ///   `null` defaults to sample - wise weights(1D). If the model has multiple outputs, you can use a different `sample_weight_mode` 
        ///     on each output by passing a dictionary or a list of modes.</param>
        /// 
        public virtual void Compile(IOptimizer optimizer, Dictionary<string, ILoss> loss, Dictionary<string, List<IMetric>> metrics = null,
            Dictionary<string, double> loss_weights = null, Dictionary<string, string> sample_weight_mode = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L681

            if (loss == null)
                loss = new Dictionary<string, ILoss>();

            this.optimizer = optimizer;
            this.sample_weight_mode = sample_weight_mode;
            this.loss = loss;

            this.loss_weights = loss_weights;

            // Prepare loss functions.
            var loss_functions = new List<ILoss>();

            if (loss.is_dict())
            {
                foreach (string name in loss.Keys)
                {
                    if (!this.output_names.Contains(name))
                        throw new Exception($"Unknown entry in loss dictionary: {name}. Only expected the following keys: {this.output_names}");
                }
                foreach (string name in this.output_names)
                {
                    if (!loss.ContainsKey(name))
                    {
                        Trace.TraceWarning($"Output {name} missing from loss dictionary. We assume this was done on purpose, and we will not be expecting any data to be passed to {name} during training.");
                        loss_functions.Add(loss[name]);
                    }
                }
            }
            else if (loss.is_list())
            {
                List<ILoss> list = loss.to_list();
                if (list.Count != this.outputs.Count)
                    throw new Exception($"When passing a list as loss, it should have one entry per model outputs. The model has " +
                        $"{this.outputs.Count} outputs, but you passed loss={loss}");
                loss_functions = list;
            }
            else
            {
                ILoss loss_function = loss.to_single();
                loss_functions = this.outputs.Select(x => loss_function).ToList();
            }

            this.loss_functions = loss_functions;
            List<ILoss> weighted_losses = loss_functions.Select(fn => _weighted_masked_objective(fn)).ToList();

            var skip_indices = new List<int>();
            this._feed_outputs = new List<Tensor>();
            this._feed_output_names = new List<string>();
            this._feed_output_shapes = new List<int?[]>();
            this._feed_loss_fns = new List<ILoss>();
            for (int i = 0; i < weighted_losses.Count; i++)
            {
                if (weighted_losses[i] == null)
                {
                    skip_indices.Add(i);
                }
                else
                {
                    this._feed_outputs.Add(this.outputs[i]);
                    this._feed_output_names.Add(this.output_names[i]);
                    this._feed_output_shapes.Add(this.internal_output_shapes[i]);
                    this._feed_loss_fns.Add(this.loss_functions[i]);
                }
            }

            // Prepare output masks.
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L774
            var masks = this.compute_mask(this.inputs, mask: null);

            if (masks == null)
                masks = this.output.Select(x => (Tensor)null).ToList();

            // Prepare loss weights.
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L781
            List<double> loss_weights_list;
            if (loss_weights == null)
            {
                loss_weights_list = this.outputs.Select(x => 1.0).ToList();
            }
            else if (loss_weights.is_dict())
            {
                foreach (string name in loss_weights.Keys)
                {
                    if (!this.output_names.Contains(name))
                        throw new InvalidOperationException($"Unknown entry in loss_weights dictionary: '{name}'. Only expected the following keys: {str(this.output_names)}.");
                }

                loss_weights_list = new List<double>();
                foreach (string name in this.output_names)
                    loss_weights_list.Add(loss_weights.get(name, 1.0));
            }
            else if (loss_weights.is_list())
            {
                List<double> lw = loss_weights.to_list();
                if (lw.Count != this.outputs.Count)
                    throw new InvalidOperationException($"When passing a list as loss_weights, it should have one entry per model outputs. The model has {str(this.outputs.Count)} outputs, but you passed loss_weights='{str(loss_weights)}'.");
                loss_weights_list = lw;
            }
            else
            {
                throw new InvalidOperationException($"Could not interpret loss_weights argument: {str(loss_weights)} - expected a list of dicts.");
            }

            // Prepare sample weights.
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L807
            var sample_weights = new List<Tensor>();
            var sample_weight_modes = new List<string>();
            if (sample_weight_mode.is_dict())
            {
                foreach (string name in sample_weight_mode.Keys)
                {
                    if (!this.output_names.Contains(name))
                    {
                        if (!this.output_names.Contains(name))
                            throw new InvalidOperationException($"Unknown entry in sample_weight_mode dictionary: {name}. Only expected the following keys: {str(this.output_names)}.");
                    }
                }

                for (int i = 0; i < this.output_names.Count; i++)
                {
                    string name = this.output_names[i];

                    Tensor weight;
                    if (skip_indices.Contains(i))
                    {
                        weight = null;
                        sample_weight_modes.Add(null);
                    }
                    else
                    {
                        if (!sample_weight_mode.Keys.Contains(name))
                            throw new InvalidOperationException($"Output '{name}' missing from sample_weight_modes dictionary.");

                        if (sample_weight_mode[name] == "temporal")
                        {
                            weight = K.placeholder(ndim: 2, name: name + "_sample_weights");
                            sample_weight_modes.Add("temporal");
                        }
                        else
                        {
                            weight = K.placeholder(ndim: 1, name: name + "_sample_weights");
                            sample_weight_modes.Add(null);
                        }
                    }
                    sample_weights.Add(weight);
                }
            }
            else if (sample_weight_mode.is_list())
            {
                var swm = sample_weight_mode.to_list();
                if (swm.Count != this.outputs.Count)
                    throw new InvalidOperationException($"When passing a list as sample_weight_mode, it should have one entry per model outputs. The model has {str(this.outputs.Count)} outputs, but you passed sample_weight_mode={str(sample_weight_mode)}");

                for (int i = 0; i < this.output_names.Count; i++)
                {
                    Tensor weight;
                    if (skip_indices.Contains(i))
                    {
                        weight = null;
                        swm.Add(null);
                    }
                    else
                    {
                        var mode = swm[i];
                        name = this.output_names[i];
                        if (mode == "temporal")
                        {
                            weight = K.placeholder(ndim: 2, name: name + "_sample_weights");
                            sample_weight_modes.Add("temporal");
                        }
                        else
                        {
                            weight = K.placeholder(ndim: 1, name: name + "_sample_weights");
                            sample_weight_modes.Add(null);
                        }
                    }

                    sample_weights.Add(weight);
                }
            }
            else
            {
                for (int i = 0; i < this.output_names.Count; i++)
                {
                    var swm = sample_weight_mode.to_single();
                    string name = this.output_names[i];

                    if (skip_indices.Contains(i))
                    {
                        sample_weight_modes.Add(null);
                        sample_weights.Add(null);
                    }
                    else
                    {
                        if (swm == "temporal")
                        {
                            sample_weights.Add(K.placeholder(ndim: 2, name: name + "_sample_weights"));
                            sample_weight_modes.Add("temporal");
                        }
                        else
                        {
                            sample_weights.Add(K.placeholder(ndim: 1, name: name + "_sample_weights"));
                            sample_weight_modes.Add(null);
                        }
                    }
                }
            }

            this.sample_weight_modes = sample_weight_modes;
            this._feed_sample_weight_modes = new List<string>();
            for (int i = 0; i < this.outputs.Count; i++)
                if (!skip_indices.Contains(i))
                    this._feed_sample_weight_modes.Add(this.sample_weight_modes[i]);

            // Prepare targets of model.
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L882
            this.targets = new List<Tensor>();
            this._feed_targets = new List<Tensor>();
            for (int i = 0; i < this.output_names.Count; i++)
            {
                string name = this.output_names[i];

                if (skip_indices.Contains(i))
                {
                    this.targets.Add(null);
                }
                else
                {
                    int?[] shape = this.internal_output_shapes[i];
                    //Tensor target = K.placeholder(ndim: shape.Length, name: name + "_target", sparse: K.is_sparse(this.outputs[i]), dtype: K.dtype(this.outputs[i]));
                    Tensor target = K.placeholder(shape: shape, ndim: shape.Length, name: name + "_target", sparse: K.is_sparse(this.outputs[i]), dtype: K.dtype(this.outputs[i]));
                    this.targets.Add(target);
                    this._feed_targets.Add(target);
                }
            }

            // Prepare metrics.
            this.metrics = metrics;
            this.metrics_names = new List<string>() { "loss" };
            this.metrics_tensors = new List<Tensor>();

            // Compute total loss.
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L903
            Tensor total_loss = null;
            using (K.name_scope("loss"))
            {
                for (int i = 0; i < this.outputs.Count; i++)
                {
                    if (skip_indices.Contains(i))
                        continue;

                    Tensor y_true = this.targets[i];
                    Tensor y_pred = this.outputs[i];
                    ILoss weighted_loss = weighted_losses[i];
                    Tensor sample_weight = sample_weights[i];
                    Tensor mask = masks[i];
                    double loss_weight = loss_weights_list[i];

                    Tensor output_loss;
                    using (K.name_scope(this.output_names[i] + "_loss"))
                        output_loss = weighted_loss.Call(y_true, y_pred, sample_weight, mask);

                    if (this.outputs.Count > 1)
                    {
                        this.metrics_tensors.Add(output_loss);
                        this.metrics_names.Add(this.output_names[i] + "_loss");
                    }

                    if (total_loss == null)
                        total_loss = K.mul(loss_weight, output_loss);
                    else
                        total_loss = K.add(total_loss, K.mul(loss_weight, output_loss));

                }

                if (total_loss == null)
                {
                    if (this.losses.Count == 0)
                        throw new Exception($"The model cannot be compiled because it has no loss to optimize.");
                    else total_loss = K.constant(0.0);
                }

                // Add regularization penalties
                // and other layer-specific losses.
                foreach (Tensor loss_tensor in this.losses)
                {
                    total_loss = K.add(total_loss, loss_tensor);
                }
            }

            // List of same size as output_names.
            // contains tuples (metrics for output, names of metrics).
            List<List<IMetric>> nested_metrics = _collect_metrics(metrics, this.output_names);

            void append_metric(int layer_num, string metric_name, Tensor metric_tensor)
            {
                // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L939
                if (this.output_names.Count > 1)
                    metric_name = this.output_layers[layer_num].name + '_' + metric_name;

                this.metrics_names.Add(metric_name);
                this.metrics_tensors.Add(metric_tensor);
            }

            for (int i = 0; i < this.outputs.Count; i++)
            {
                if (skip_indices.Contains(i))
                    continue;

                Tensor y_true = this.targets[i];
                Tensor y_pred = this.outputs[i];
                List<IMetric> output_metrics = nested_metrics[i];

                foreach (IMetric metric in output_metrics)
                {
                    if (metric is Accuracy)
                    {
                        IMetric acc_fn;

                        // custom handling of accuracy
                        // (because of class mode duality)
                        int?[] output_shape = this.internal_output_shapes[i];

                        if (output_shape.Get(-1) == 1 || this.loss_functions[i] is BinaryCrossEntropy)
                        {
                            // case: binary accuracy
                            acc_fn = new BinaryAccuracy();
                        }
                        else if (this.loss_functions[i] is SparseCategoricalCrossEntropy)
                        {
                            // case: categorical accuracy with sparse targets
                            acc_fn = new SparseCategoricalAccuracy();
                        }
                        else
                        {
                            acc_fn = new CategoricalAccuracy();
                        }

                        var masked_fn = _masked_objective(acc_fn);
                        append_metric(i, "acc", masked_fn.Call(y_true, y_pred, mask: masks[i]));
                    }
                    else
                    {
                        // Different from the original Keras implementation:
                        var masked_metric_fn = _masked_objective(metric);
                        var metric_result = masked_metric_fn.Call(y_true, y_pred, mask: masks[i]);
                        append_metric(i, metrics_names[i], metric_result);
                    }
                }
            }

            // Prepare gradient updates and state updates.
            this.total_loss = total_loss;
            this.sample_weights = sample_weights;
            this._feed_sample_weights = new List<Tensor>();
            for (int i = 0; i < this.sample_weights.Count; i++)
            {
                if (!skip_indices.Contains(i))
                    this._feed_sample_weights.Add(sample_weights[i]);
            }

            // Functions for train, test and predict will
            // be compiled lazily when required.
            // This saves time when the user != using all functions.
            //this._function_kwargs = kwargs

            this.train_function = null;
            this.test_function = null;
            this.predict_function = null;

            // Collected trainable weights and sort them deterministically.
            trainable_weights = this.trainable_weights;

            // Sort weights by name.
            if (trainable_weights.Count > 0)
            {
                trainable_weights.OrderBy(keySelector: x => x.name);
                this._collected_trainable_weights = trainable_weights;
            }
        }



        /// <summary>
        ///   Maps metric functions to model outputs.
        /// </summary>
        /// 
        /// <param name="metrics">A list or dict of metric functions.</param>
        /// <param name="output_names">A list of the names (strings) of model outputs.</param>
        /// 
        /// <returns>A list (one entry per model output) of lists of metric functions.</returns>
        /// 
        private List<List<IMetric>> _collect_metrics(Dictionary<string, List<IMetric>> metrics, List<string> output_names)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L293

            if (metrics == null || (metrics.Count == 1 && metrics.Values.First() == null))
                return output_names.Select(x => new List<IMetric>()).ToList();

            if (metrics.is_single())
            {
                // we then apply all metrics to all outputs.
                return output_names.Select(x => metrics.to_single()).ToList();
            }
            else if (metrics.is_dict())
            {
                var nested_metrics = new List<List<IMetric>>();
                foreach (string name in output_names)
                {
                    List<IMetric> output_metrics = metrics.get(name, new List<IMetric>());
                    nested_metrics.Add(output_metrics);
                }

                return nested_metrics;
            }

            throw new InvalidOperationException($"Type of `metrics` argument not understood. Expected a list or dictionary, found: {str(metrics)}.");
        }

        /// <summary>
        ///   Adds support for masking to an objective function.
        /// </summary>
        /// 
        /// <remarks>
        ///   It transforms an objective function <c>fn(y_true, y_pred)</c> into a cost-masked objective 
        ///   function </c>fn(y_true, y_pred, mask)</c>.
        /// </remarks>
        /// 
        /// <param name="metric_fn">The objective function to wrap, with signature <c>fn(y_true, y_pred)</c>.</param>
        /// 
        /// <returns>A function with signature <c>fn(y_true, y_pred, mask)</c>.</returns>
        /// 
        private IMetric _masked_objective(IMetric metric_fn)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L459

            Tensor masked(Tensor y_true, Tensor y_pred, Tensor mask = null)
            {
                // score_array has ndim >= 2
                Tensor score_array = metric_fn.Call(y_true, y_pred);

                if (mask != null)
                {
                    // Cast the mask to floatX to avoid float64 upcasting in theano
                    mask = K.cast(mask, K.floatx());
                    // mask should have the same shape as score_array
                    score_array *= mask;
                    //  the loss per batch should be proportional
                    //  to the number of unmasked samples.
                    score_array /= K.mean(mask);
                }
                return K.mean(score_array);
            }

            return new CustomMetric(masked);
        }

        /// <summary>
        ///   Adds support for masking and sample-weighting to an objective function.
        ///   It transforms an objective function `fn(y_true, y_pred)` into a sample - 
        ///   weighted, cost - masked objective function `fn(y_true, y_pred, weights, mask)`.
        /// </summary>
        /// 
        /// <param name="fn">The objective function to wrap, with signature `fn(y_true, y_pred)`.</param>
        /// 
        /// <returns>A function with signature `fn(y_true, y_pred, weights, mask)`.</returns>
        /// 
        private ILoss _weighted_masked_objective(ILoss fn)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L406

            if (fn == null)
                return null;

            Tensor weighted(Tensor y_true, Tensor y_pred, Tensor weights, Tensor mask = null)
            {
                // score_array has ndim >= 2
                Tensor score_array = fn.Call(y_true, y_pred);

                if (mask != null)
                {
                    // Cast the mask to floatX to avoid float64 upcasting in theano
                    mask = K.cast(mask, K.floatx());
                    // mask should have the same shape as score_array
                    score_array = score_array * mask;
                    //  the loss per batch should be proportional
                    //  to the number of unmasked samples.
                    score_array = score_array / K.mean(mask);
                }
                // reduce score_array to same ndim as weight array
                int? ndim = K.ndim(score_array);
                int? weight_ndim = K.ndim(weights);
                if (ndim != weight_ndim)
                    score_array = K.mean(score_array, axis: range(weight_ndim, ndim));

                // apply sample weighting
                if (weights != null)
                {
                    score_array = score_array * weights;
                    score_array = score_array / K.mean(K.cast(K.not_equal(weights, 0), K.floatx()));
                }
                return K.mean(score_array);
            }

            return new CustomLoss(weighted);
        }



        public void _make_train_function()
        {
            if (this.train_function == null)
            {
                var inputs = (Enumerable.Concat(this._feed_inputs, this._feed_targets).Concat(this._feed_sample_weights)).ToList();

                if (this.uses_learning_phase && !(K.learning_phase() is int))
                    inputs.Add((Tensor)K.learning_phase());

                List<List<Tensor>> training_updates = this.optimizer.get_updates(this._collected_trainable_weights, this.constraints, this.total_loss);
                List<List<Tensor>> updates = Enumerable.Concat(this.updates, training_updates).ToList();

                // Gets loss and metrics. Updates weights at each call.	
                this.train_function = K.function(inputs, ((new[] { this.total_loss }).Concat(this.metrics_tensors)).ToList(),
                    updates: updates, name: "train_function"); //, **this._function_kwargs)
            }
        }


        public void _make_test_function()
        {
            if (this.test_function == null)
                throw new Exception("You must compile your model before using it.");

            if (this.test_function == null)
            {
                var inputs = this._feed_inputs.Concat(this._feed_targets).Concat(this._feed_sample_weights).ToList();
                if (this.uses_learning_phase && !(K.learning_phase() is int))
                    inputs.Add((Tensor)K.learning_phase());

                // Return loss and metrics, no gradient updates.
                // Does update the network states.
                this.test_function = K.function(inputs, new[] { this.total_loss }.Concat(this.metrics_tensors).ToList(),
                    updates: this.state_updates(), name: "test_function"); //, **this._function_kwargs);
            }
        }

        public void _make_predict_function()
        {
            if (this.predict_function == null)
            {
                if (this.uses_learning_phase && !(K.learning_phase() is int))
                {
                    inputs = this._feed_inputs.Concat(new List<Tensor>() { (Tensor)K.learning_phase() }).ToList();
                }
                else
                {
                    inputs = this._feed_inputs;
                }
                // Gets network outputs. Does not update weights.
                // Does update the network states.
                // kwargs = getattr( "_function_kwargs', { });

                this.predict_function = K.function(inputs, this.outputs,
                    updates: this.state_updates(), name: "predict_function"); //, **kwargs);
            }
        }



        public History _fit_loop(Function f, List<Array> ins, List<string> out_labels = null,
            int batch_size = 32, int epochs = 100, int verbose = 1, CallbackList callbacks = null,
                      Function val_f = null, List<Array> val_ins = null, Shuffle shuffle = Shuffle.True,
                      List<String> callback_metrics = null, int initial_epoch = 0)
        {
            // """Abstract fit function for `f(ins)`.
            // Assume that f returns a list, labeled by out_labels.
            // // Arguments
            // f: Keras function returning a list of tensors
            // ins: list of tensors to be fed to `f`
            // out_labels: list of strings, display names of
            // the outputs of `f`
            // batch_size: integer batch size
            // epochs: number of times to iterate over the data
            // verbose: verbosity mode, 0, 1 or 2
            // callbacks: list of callbacks to be called during training
            // val_f: Keras function to call for validation
            // val_ins: list of tensors to be fed to `val_f`
            // shuffle: whether to shuffle the data at the beginning of each epoch
            // callback_metrics: list of strings, the display names of the metrics
            // passed to the callbacks. They should be the
            // concatenation of list the display names of the outputs of
            // `f` and the list of display names of the outputs of `f_val`.
            // initial_epoch: epoch at which to start training
            // (useful for resuming a previous training run)
            // // Returns
            // `History` object.
            // """
            bool do_validation = false;
            if (val_f != null && val_ins != null)
                do_validation = true;

            if (verbose > 0)
                Trace.Write("Train on {ins[0].shape[0]} samples, validate on val_ins[0].shape[0] samples");

            int num_train_samples;
            if (ins != null)
            {
                num_train_samples = ins[0].GetLength(0);
            }
            else
            {
                // May happen if we are running `fit` without Numpy input data,
                // i.e. if all inputs to the models are data tensors
                // instead of placeholders.
                // In that case we will run `fit` over a single batch.
                num_train_samples = batch_size;
                verbose = 2;
            }

            int[] index_array = Vector.Range(num_train_samples);

            this.history = new History();

            if (callbacks == null)
                callbacks = new CallbackList();
            callbacks.Add(new BaseLogger());
            callbacks.Add(this.history);

            if (verbose > 0)
                callbacks.Add(new ProgbarLogger());

            //callbacks = cbks.CallbackList(callbacks);

            if (out_labels == null)
                out_labels = new List<string>();

            // it's possible to callback a different model than this
            // (used by Sequential models)
            Model callback_model;
            if (this.callback_model != null && this.callback_model != null)
                callback_model = this.callback_model;
            else
                callback_model = this;

            callbacks.set_model(callback_model);
            callbacks.set_params(new Dictionary<string, object> {
                {"batch_size", batch_size },
                {"epochs", epochs },
                {"samples", num_train_samples },
                {"verbose", verbose },
                {"do_validation", do_validation },
                { "metrics", callback_metrics },
            });

            callbacks.on_train_begin();
            callback_model.stop_training = false;
            foreach (Callback cbk in callbacks)
                cbk.validation_data = val_ins;

            for (int epoch = initial_epoch; epoch < epochs; epoch++)
            {
                callbacks.on_epoch_begin(epoch);

                if (shuffle == Shuffle.Batch)
                    index_array = _batch_shuffle(index_array, batch_size);

                else if (shuffle == Shuffle.True)
                    Vector.Shuffle(index_array);

                List<(int, int)> batches = _make_batches(num_train_samples, batch_size);
                var epoch_logs = new Dictionary<string, object>();

                for (int batch_index = 0; batch_index < batches.Count; batch_index++)
                {
                    var (batch_start, batch_end) = batches[batch_index];
                    int[] batch_ids = index_array.Get(batch_start, batch_end);

                    List<Array> ins_batch;
                    //try
                    //{
                    //if (ins[-1] is float)
                    //{
                    //    // Do not slice the training phase flag.
                    //    ins_batch = _slice_arrays(ins.Get(0, -1), batch_ids) .Concatenate( [ins[-1]];
                    //}
                    //else
                    //{
                    ins_batch = _slice_arrays(ins, batch_ids);
                    //}
                    //}
                    //catch
                    //{
                    //    throw new Exception($"TypeError while preparing batch. If using HDF5 input data, pass shuffle='batch'.");
                    //}

                    var batch_logs = new Dictionary<string, object>();
                    batch_logs["batch"] = batch_index;
                    batch_logs["size"] = batch_ids.Length;
                    callbacks.on_batch_begin(batch_index, batch_logs);
                    List<Tensor> outs = f.Call(ins_batch);

                    for (int i = 0; i < out_labels.Count; i++)
                    {
                        var l = out_labels[i];
                        var o = outs[i];
                        batch_logs[l] = o;
                    }

                    callbacks.on_batch_end(batch_index, batch_logs);
                    if (callback_model.stop_training)
                        break;

                    if (batch_index == batches.Count - 1)  // Last batch.
                    {
                        if (do_validation)
                        {
                            List<Tensor> val_outs = this._test_loop(val_f, val_ins, batch_size: batch_size, verbose: 0);

                            // Same labels assumed.
                            for (int i = 0; i < out_labels.Count; i++)
                            {
                                var l = out_labels[i];
                                var o = val_outs[i];
                                epoch_logs["val_" + l] = o;
                            }
                        }
                    }

                    callbacks.on_epoch_end(epoch, epoch_logs);

                    if (callback_model.stop_training)
                        break;
                }
            }

            callbacks.on_train_end();
            return this.history;
        }

        /// <summary>
        ///   Slice an array or list of arrays.
        /// </summary>
        /// 
        /// <remarks>
        ///  This takes an array-like, or a list of array-likes, and outputs:
        /// - arrays[start:stop] if `arrays` is an array-like
        /// - [x[start: stop] for x in arrays] if `arrays` is a list
        /// Can also work on list/array of indices: `_slice_arrays(x, indices)`
        /// </remarks>
        /// 
        /// <param name="arrays">Single array or list of arrays.</param>
        /// <param name="batch_ids">The batch ids.</param>
        /// <returns>List&lt;Array&gt;.</returns>
        /// <exception cref="NotImplementedException"></exception>
        private List<Array> _slice_arrays(List<Array> arrays, int[] start = null, int? stop = null)
        {
            if (arrays == null)
                return new List<Array>() { null };

            if (start.Length > 1 && stop != null)
                throw new Exception();

            var r = new List<Array>(arrays.Count);
            for (int i = 0; i < arrays.Count; i++)
            {
                if (stop != null)
                    throw new NotSupportedException();
                else
                    r.Add(Accord.Math.Matrix.Get(arrays[i], dimension: 0, indices: start));
            }

            return r;
        }

        /// <summary>
        ///   Returns a list of batch indices (tuples of indices).
        /// </summary>
        /// <param name="size">Integer, total size of the data to slice into batches.</param>
        /// <param name="batch_size">Integer, batch size.</param>
        /// 
        /// <returns>A list of tuples of array indices.</returns>
        /// 
        private List<(int, int)> _make_batches(int size, int batch_size)
        {
            // https://github.com/fchollet/keras/blob/8ac788a5614570d222c484ab49cf9e878eeab9ff/keras/engine/training.py#L364

            int num_batches = (int)(Math.Ceiling(size / (double)(batch_size)));
            return Vector.Range(0, num_batches).Select(i => (i * batch_size, Math.Min(size, (i + 1) * batch_size))).ToList();
        }

        private int[] _batch_shuffle(int[] index_array, int batch_size)
        {
            throw new NotImplementedException();
        }




        public Array[] _predict_loop(Function f, List<Array> ins, int batch_size = 32, int verbose = 0)
        {
            // """Abstract method to loop over some data in batches.
            // // Arguments
            // f: Keras function returning a list of tensors.
            // ins: list of tensors to be fed to `f`.
            // batch_size: integer batch size.
            // verbose: verbosity mode.
            // // Returns
            // Array of predictions (if the model has a single output)
            // or list of arrays of predictions
            // (if the model has multiple outputs).
            // """
            int samples;
            if (ins != null)
            {
                samples = ins[0].GetLength(0);
            }
            else
            {
                // May happen if we are running `predict` without Numpy input data,
                // i.e. if all inputs to the models are data tensors
                // instead of placeholders.
                // In that case we will run `predict` over a single batch.
                samples = batch_size;
                verbose = 2;
            }

            var outs = new List<Array>();

            Progbar progbar = null;
            if (verbose == 1)
                progbar = new Progbar(target: samples);

            List<(int, int)> batches = _make_batches(samples, batch_size);

            int[] index_array = Vector.Range(samples);


            for (int batch_index = 0; batch_index < batches.Count; batch_index++)
            {
                var (batch_start, batch_end) = batches[batch_index];

                var batch_ids = index_array.Get(batch_start, batch_end);
                //if (ins != null && ins[-1] is float)
                //{
                //    // Do not slice the training phase flag.
                //    ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]];
                //}
                //else
                //{
                List<Array> ins_batch = _slice_arrays(ins, batch_ids);
                //}

                List<Tensor> batch_outs = f.Call(ins_batch);


                if (batch_index == 0)
                {
                    foreach (var batch_out in batch_outs)
                    {
                        var shape = new int[] { samples }.Concatenate(batch_out.shape.Apply(x=>x.Value).Get(1, 0));
                        outs.Add(Matrix.Zeros(batch_out.dtype.Value.ToType(), shape));
                    }
                }

                for (int i = 0; i < batch_outs.Count; i++)
                {
                    var batch_out = batch_outs[i];
                    Array array = (Array)batch_out.eval();
                    Accord.Math.Matrix.Set(outs[i], dimension: 0, start: batch_start, end: batch_end, value: array);
                    if (verbose == 1)
                        progbar.update(batch_end);
                }
            }

            return outs.ToArray();
        }

        /// <summary>
        ///   Abstract method to loop over some data in batches.
        /// </summary>
        /// 
        /// <param name="f">Function returning a list of tensors.</param>
        /// <param name="ins">The list of tensors to be fed to `f`.</param>
        /// <param name="batch_size">The batch size.</param>
        /// <param name="verbose">The verbosity mode.</param>
        /// <returns>
        ///    Scalar loss (if the model has a single output and no metrics)
        ///    or list of scalars (if the model has multiple outputs
        ///    and/or metrics). The attribute `model.metrics_names` will give you
        ///    the display labels for the scalar outputs.
        /// </returns>
        /// 
        public List<Tensor> _test_loop(Function f, List<Array> ins, int batch_size = 32, int verbose = 0)
        {
            int samples;
            if (ins != null)
            {
                samples = ins[0].GetLength(0);
            }
            else
            {
                // May happen if we are running `evaluate` without Numpy input data,
                // i.e. if all inputs to the models are data tensors
                // instead of placeholders.
                // In that case we will run `evaluate` over a single batch.
                samples = batch_size;
                verbose = 2;
            }

            var outs = new List<Tensor>();
            Progbar progbar = null;
            if (verbose == 1)
                progbar = new Progbar(target: samples);

            var batches = _make_batches(samples, batch_size);
            int[] index_array = Vector.Range(samples);

            for (int batch_index = 0; batch_index < batches.Count; batch_index++)
            {
                var (batch_start, batch_end) = batches[batch_index];

                var batch_ids = index_array.Get(batch_start, batch_end);

                //if (ins[-1] is float)
                //{
                //    // Do not slice the training phase flag.
                //    ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]];
                //}
                //else
                //{
                var ins_batch = _slice_arrays(ins, batch_ids);
                //}

                var batch_outs = f.Call(ins_batch);
                if (batch_index == 0)
                {
                    foreach (Tensor batch_out in batch_outs)
                        outs.Add(K.constant(0.0));

                    for (int i = 0; i < batch_outs.Count; i++)
                    {
                        Tensor batch_out = batch_outs[i];
                        outs[i] = K.add(outs[i], K.mul(batch_out, batch_ids.Length));
                    }
                }
                else
                {
                    if (batch_index == 0)
                        outs.Add(K.constant(0.0));
                    outs[0] = K.add(outs[0], K.mul(batch_outs, batch_ids.Length));
                }

                if (verbose == 1)
                    progbar.update(batch_end);
            }

            for (int i = 0; i < outs.Count; i++)
                outs[i] = K.div(outs[i], samples);

            //return outs;
            return null;
        }

        private object _slice_arrays(List<Tensor> ins, int[] batch_ids)
        {
            throw new NotImplementedException();
        }

        public (List<Array>, List<Array>, List<Array>) _standardize_user_data(Dictionary<string, Array> x, Dictionary<string, Array> y,
                                   Dictionary<string, Array> sample_weight = null, Dictionary<string, Dictionary<string, double>> class_weight = null,
                                   bool check_batch_axis = true, int? batch_size = null)
        {
            if (this.optimizer == null)
                throw new Exception("You must compile a model before training/testing. Use `model.compile(optimizer, loss)`.");

            var output_shapes = new List<int?[]>();
            for (int i = 0; i < this._feed_output_shapes.Count; i++)
            {
                var output_shape = this._feed_output_shapes[i];
                var loss_fn = this._feed_loss_fns[i];

                if (loss_fn is SparseCategoricalCrossEntropy)
                    output_shapes.Add(output_shape.Get(0, -1).Concat(new int?[] { 1 }).ToArray());
                else
                    output_shapes.Add(output_shape);
            }

            var xx = _standardize_input_data(x, this._feed_input_names, this._feed_input_shapes, check_batch_axis: false, exception_prefix: "input");
            var yy = _standardize_input_data(y, this._feed_output_names, output_shapes, check_batch_axis: false, exception_prefix: "target");

            List<Array> sample_weights = _standardize_sample_weights(sample_weight, this._feed_output_names).to_list();
            List<Dictionary<string, double>> class_weights = _standardize_class_weights(class_weight, this._feed_output_names).to_list();

            sample_weights = Vector.Range(_feed_sample_weight_modes.Count).Select(i =>
                (Array)_standardize_weights(yy[i], sample_weights[i], class_weights[i], this._feed_sample_weight_modes[i])).ToList();

            _check_array_lengths(xx, yy, sample_weights);
            _check_loss_and_target_compatibility(yy, this._feed_loss_fns, this._feed_output_shapes);
            if (this.stateful && batch_size > 0)
            {
                if (xx[0].GetLength(0) % batch_size != 0)
                    throw new Exception($"In a stateful network, you should only pass inputs with a number of samples that can be divided by the batch size. Found: {xx[0].GetLength(0)} samples");
            }

            return (xx, yy, sample_weights);
        }

        private Dictionary<string, Dictionary<string, double>> _standardize_class_weights(Dictionary<string, Dictionary<string, double>> class_weight, List<string> feed_output_names)
        {
            //https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L190
            return _standardize_sample_or_class_weights(class_weight, output_names, "class_weight");
        }

        private double[] _standardize_weights(Array y, Array sample_weight, Dictionary<string, double> class_weight, string sample_weight_mode)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L499
            if (sample_weight_mode != null)
            {
                if (sample_weight_mode != "temporal")
                    throw new ArgumentException($"sample_weight_mode should be None or 'temporal'. Found: {str(sample_weight_mode)}");
                if (y.Rank < 3)
                    throw new ArgumentException("Found a sample_weight array for an input with shape {str(y.shape)}. Timestep-wise sample weighting (use of sample_weight_mode='temporal') is restricted to outputs that are at least 3D, i.e. that have a time dimension.");
                if (sample_weight != null && sample_weight.Rank != 2)
                    throw new ArgumentException($"Found a sample_weight array with shape {str(sample_weight.GetLength())}. In order to use timestep-wise sample weighting, you should pass a 2D sample_weight array.");
            }
            else
            {
                if (sample_weight != null && sample_weight.Rank != 1)
                    throw new ArgumentException($"Found a sample_weight array with shape {str(sample_weight.GetLength())}. In order to use timestep - wise sample weights, you should specify sample_weight_mode = 'temporal' in compile().If you just mean to use sample - wise weights, make sure your sample_weight array is 1D.");
            }

            if (sample_weight != null)
            {
                if (sample_weight.Rank > y.Rank)
                    throw new ArgumentException($"Found a sample_weight with shape {str(sample_weight.GetLength())}. Expected sample_weight with rank less than or equal to {y.Rank}.");

                if (!y.GetLength().Get(0, sample_weight.Rank).IsEqual(sample_weight.GetLength()))
                    throw new ArgumentException($"Found a sample_weight array with shape {str(sample_weight.GetLength())} for an input with shape {str(y.GetLength())}. sample_weight cannot be broadcast.");
                return (double[])sample_weight;
            }

            else if (class_weight.is_dict())
            {
                if (y.Rank > 2)
                    throw new ArgumentException($"class_weight not supported for 3+ dimensional targets.");

                int[] y_classes;
                if (y.GetLength(1) > 1)
                    y_classes = (y as double[,]).ArgMax(dimension: 1);
                else if (y.GetLength(1) == 1)
                    y_classes = (int[])y.Squeeze();
                else
                    y_classes = (y as int[]);
                double[] weights = y_classes.Select(cls => class_weight[cls.ToString()]).ToArray();
                return weights;
            }
            else
            {
                if (sample_weight_mode == null)
                {
                    // TODO: Should use Vector.Ones
                    return Vector.Create(y.GetLength(0), (double)1);
                }
                else
                {
                    throw new NotImplementedException();
                    //return Matrix.Ones<double?>(y.GetLength(0), y.GetLength(1)));
                }
            }
        }

        private Dictionary<string, Array> _standardize_sample_weights(Dictionary<string, Array> sample_weight, List<string> output_names)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L196
            return _standardize_sample_or_class_weights(sample_weight, output_names, "sample_weight");
        }

        private Dictionary<string, T> _standardize_sample_or_class_weights<T>(Dictionary<string, T> x_weight, List<string> output_names, string weight_type)
            where T : class
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L143
            if (x_weight == null || x_weight.Count == 0)
                return output_names.Select(x => (T)null).ToList().dict_from_list();

            if (output_names.Count == 1)
            {
                if (x_weight.is_list() && x_weight.Values.Count == 1)
                    return x_weight.to_list().dict_from_list();
                if (x_weight.is_dict() && x_weight.ContainsKey(output_names[0]))
                    return new List<T>() { x_weight[output_names[0]] }.dict_from_list();
                return new List<T>() { x_weight.to_single() }.dict_from_list();
            }

            if (x_weight.is_list())
            {
                List<T> lx_weight = x_weight.to_list();
                if (lx_weight.Count != output_names.Count())
                {
                    throw new ArgumentException($"Provided {weight_type} was a list of {str(x_weight.Count)} elements, but the model has {str(output_names.Count)} outputs. You should provide one {weight_type} array per model output.");
                }
                return lx_weight.dict_from_list();
            }

            if (x_weight.is_dict())
            {
                var x_weights = new List<T>();
                foreach (string name in output_names)
                    x_weights.Add(x_weight[name]);
                return x_weights.dict_from_list();
            }
            else
            {
                throw new ArgumentException($"The model has multiple outputs, so {weight_type} should be either a list of a dict. Provided {weight_type} type not understood: {str(x_weight)}.");
            }
        }

        private double _standardize_weights(Array yy, Tensor sw, Tensor cw, object mode)
        {
            throw new NotImplementedException();
        }

        private object _standardize_class_weights(Dictionary<int, Tensor> class_weight, List<string> feed_output_names)
        {
            throw new NotImplementedException();
        }

        private List<Tensor> _standardize_sample_weights(List<Tensor> sample_weight, List<string> feed_output_names)
        {
            throw new NotImplementedException();
        }

        private List<Array> _standardize_input_data(Dictionary<string, Array> data, List<string> names, List<int?[]> shapes, bool check_batch_axis = true, string exception_prefix = "")
        {
            if (data == null)
                return names.Select(x => (Array)null).ToList();

            List<Array> arrays;

            if (data.is_dict())
            {
                arrays = new List<Array>();
                foreach (string name in names)
                {
                    if (!data.ContainsKey(name))
                        throw new ArgumentException($"No data provided for {name}. Need data for each key in: {str(names)}");

                    arrays.Add(data[name]);
                }
            }
            else if (data.is_list())
            {
                List<Array> ldata = data.to_list();

                if (ldata.Count != names.Count)
                {
                    if (data != null)
                    {
                        throw new ArgumentException($"Error when checking model {exception_prefix}: the list of Numpy arrays that you are passing to your model " +
                            $"is not the size the model expected. Expected to see {str(names.Count)} array(s), but instead got the following list of ' + {str(ldata.Count)} " +
                            $"arrays: {str(data).Substring(0, 200)} ...");
                    }
                    else
                    {
                        if (names.Count != 1)
                        {
                            throw new ArgumentException($"Error when checking model {exception_prefix}: you are passing a list as input to your model, but the model " +
                                $"expects a list of {str(names.Count)} arrays instead. The list you passed was: {str(ldata).Substring(0, 200)} ...");
                        }
                    }
                }
                arrays = new List<Array>();
                arrays.AddRange(ldata);
            }
            else
            {
                Array sdata = data.to_single();

                if (names.Count > 1)
                {
                    // Case: model expects multiple inputs but only received
                    // a single Numpy array.
                    throw new ArgumentException($"The model expects {str(names.Count)} {exception_prefix} arrays, but only received one array. Found: array with shape {str(sdata.GetLength())}");
                }
                arrays = new List<Array>() { sdata };
            }

            // Make arrays at least 2D.
            for (int i = 0; i < names.Count; i++)
            {
                Array array = arrays[i];
                if (array.Rank == 1)
                {
                    array = array.ExpandDimensions(dimension: 1);
                    arrays[i] = array;
                }
            }

            // Check shapes compatibility.
            if (shapes != null)
            {
                for (int i = 0; i < names.Count; i++)
                {
                    if (shapes[i] == null)
                        continue;

                    Array array = arrays[i];

                    if (array.Rank != shapes[i].Length)
                    {
                        throw new ArgumentException($"Error when checking {exception_prefix}: expected {names[i]} to have {str(shapes[i].Length)} dimensions, but got array with shape {str(array.GetLength())}");
                    }

                    for (int j = 0; j < shapes.Count; j++)
                    {
                        int dim = array.GetLength(j);
                        int? ref_dim = shapes[i][j];

                        if (j == 0 && !check_batch_axis)
                        {
                            // skip the first axis
                            continue;
                        }

                        if (ref_dim.HasValue)
                        {
                            if (ref_dim != dim)
                            {
                                throw new ArgumentException($"Error when checking {exception_prefix}: expected {names[i]} to have shape {str(shapes[i])} but got array with shape {str(array.GetLength())}.");
                            }
                        }
                    }
                }
            }

            return arrays;
        }

        /// <summary>
        ///   Does validation on the compatibility of targets and loss functions.
        /// </summary>
        /// 
        private void _check_loss_and_target_compatibility(List<Array> targets, List<ILoss> loss_fns, List<int?[]> output_shapes)
        {
            for (int i = 0; i < targets.Count; i++)
            {
                Array y = targets[i];
                ILoss loss = loss_fns[i];
                int?[] shape = output_shapes[i];

                if (loss == null)
                    continue;

                if (loss is CategoricalCrossEntropy)
                {
                    if (y.GetLength().Get(-1) == 1)
                    {
                        throw new ArgumentException(
                            $"You are passing a target array of shape {str(y.GetLength())} while using as loss `categorical_crossentropy`. `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes). If your targets are integer classes, you can convert them to the expected format via:" +
@"
```
from keras.utils.np_utils import to_categorical
y_binary = to_categorical(y_int)
```
" +
                            "Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.");
                    }
                }

                if (loss is MeanSquareError || loss is BinaryCrossEntropy || loss is CategoricalCrossEntropy)
                {
                    for (int j = 1; j < shape.Length; j++)
                    {
                        int target_dim = y.GetLength(j);
                        int? out_dim = shape[j];
                        if (out_dim != null && target_dim != out_dim)
                            throw new ArgumentException($"A target array with shape {str(y.GetLength())} was passed for an output of shape {str(shape)} while using as loss `{loss}`. This loss expects targets to have the same shape as the output.");
                    }
                }
            }
        }

        /// <summary>
        ///   Does user input validation for arrays.
        /// </summary>
        /// 
        private void _check_array_lengths(List<Array> x, List<Array> y, List<Array> w)
        {
            int[] x_lengths = x.Select(s => s.GetLength(0)).ToArray();
            int[] y_lengths = y.Select(s => s.GetLength(0)).ToArray();
            int[] w_lengths = w.Select(s => s.GetLength(0)).ToArray();

            var set_x = set(x_lengths);
            if (len(set_x) > 1)
                throw new ArgumentException($"All input arrays (x) should have the same number of samples. Got array shapes: {str(x.Select(s => s.GetLength()))}");

            var set_y = set(y_lengths);
            if (len(set_y) > 1)
                throw new ArgumentException($"All target arrays (y) should have the same number of samples. Got array shapes: {str(y.Select(s => s.GetLength()))}");

            var set_w = set(w_lengths);
            if (len(set_w) > 1)
                throw new ArgumentException($"All sample_weight arrays should have the same number of samples. Got array shapes: {str(w.Select(s => s.GetLength()))}");

            if (set_x.Count > 0 && set_y.Count > 0 && list(set_x)[0] != list(set_y)[0])
                throw new ArgumentException($"Input arrays should have the same number of samples as target arrays. Found {str(list(set_x)[0])} input samples and {str(list(set_y)[0])} target samples.");

            if (set_y.Count > 0 && set_w.Count > 0 && list(set_y)[0] != list(set_w)[0])
                throw new ArgumentException($"Sample_weight arrays should have the same number of samples as target arrays. Got {str(list(set_y)[0])} input samples and {str(list(set_w)[0])} target samples.");
        }

        public List<String> _get_deduped_metrics_names()
        {
            var out_labels = this.metrics_names;

            // Rename duplicated metrics name
            // (can happen with an output layer shared among multiple dataflows).
            var deduped_out_labels = new List<string>();
            for (int i = 0; i < out_labels.Count; i++)
            {
                string label = out_labels[i];
                string new_label = label;
                if (out_labels.Count(x => x == label) > 1)
                {
                    int dup_idx = out_labels.ToArray().Get(0, i).Count(x => x == label);
                    new_label += "_" + (dup_idx + 1).ToString();
                }
                deduped_out_labels.Add(new_label);
            }
            return deduped_out_labels;
        }

        public History fit(
                       Array x = null,
                       Array y = null,
                       int batch_size = 32,
                       int epochs = 1,
                       int verbose = 1,
                       CallbackList callbacks = null,
                       double validation_split = 0.0,
                       IList<Array> validation_data = null,
                       bool shuffle = true,
                       Dictionary<int, double> class_weight = null,
                       Array sample_weight = null,
                       int initial_epoch = 0,
                       object kwargs = null)
        {
            return fit(x: x.dict_from_single(),
                       y: y.dict_from_single(),
                       batch_size: batch_size,
                       epochs: epochs,
                       verbose: verbose,
                       callbacks: callbacks,
                       validation_split: validation_split,
                       validation_data: validation_data.Select(a => a.dict_from_single()).ToList(),
                       shuffle: shuffle ? Shuffle.True : Shuffle.False,
                       class_weight: class_weight.Select(p => (p.Key.ToString(), p.Value)).ToDictionary(a => a.Item1, b => b.Item2).dict_from_single(),
                       sample_weight: sample_weight.dict_from_single(),
                       initial_epoch: initial_epoch,
                       kwargs: kwargs);

        }

        /// <summary>
        ///   Trains the model for a fixed number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="x">The Numpy array of training data, or list of Numpy arrays if the model has multiple inputs. If all
        ///     inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">The Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all 
        ///     outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="batch_size">The number of samples per gradient update.</param>
        /// <param name="epochs">The number of times to iterate over the training data arrays.</param>
        /// <param name="verbose">The verbosity mode: 0, 1, or 2. In which 0 = silent, 1 = verbose, 2 = one log line per epoch.</param>
        /// <param name="callbacks">The list of callbacks to be called during training.</param>
        /// <param name="validation_split">A float between 0 and 1: the fraction of the training data to be used as validation data.
        ///     The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any 
        ///     model metrics on this data at the end of each epoch.</param>
        /// <param name="validation_data">The validation  data on which to evaluate the loss and any model metrics at the end of 
        ///   each epoch. The model will not be trained on this data. This could be a tuple (x_val, y_val) or a tuple (x_val, y_val, 
        ///   val_sample_weights).</param>
        /// <param name="shuffle">whether to shuffle the training data before each epoch.</param>
        /// <param name="class_weight">The class weight. Optional dictionary mapping class indices (integers) to a weight (float) 
        ///   to apply to the model's loss for the samples from this class during training. This can be useful to tell the model 
        ///   to "pay more attention" to samples from an under-represented class.</param>
        /// <param name="sample_weight">The sample weight. Optional array of the same length as x, containing weights to apply to 
        ///   the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length),
        ///   to apply a different weight to every timestep of every sample. In this case you should make sure to specify 
        ///   sample_weight_mode="temporal" in compile().</param>
        /// <param name="initial_epoch">The initial epoch at which to start training (useful for resuming a previous training run).</param>
        /// 
        /// <returns> 
        ///   A `History` instance. Its `history` attribute contains all information collected during training.
        /// </returns>
        /// 
        public virtual History fit(
                    Dictionary<string, Array> x = null,
                    Dictionary<string, Array> y = null,
                    int batch_size = 32,
                    int epochs = 1,
                    int verbose = 1,
                    CallbackList callbacks = null,
                    double validation_split = 0.0,
                    IList<Dictionary<string, Array>> validation_data = null,
                    Shuffle shuffle = Shuffle.True,
                    Dictionary<string, Dictionary<string, double>> class_weight = null,
                    Dictionary<string, Array> sample_weight = null,
                    int initial_epoch = 0,
                    object kwargs = null)
        {
            // Validate user data.
            (List<Array> xx, List<Array> yy, List<Array> sample_weightsx) = this._standardize_user_data(
                x, y,
                sample_weight: sample_weight,
                class_weight: class_weight,
                check_batch_axis: false,
                batch_size: batch_size);

            Dictionary<string, Array> val_x = null;
            Dictionary<string, Array> val_y = null;
            Dictionary<string, Array> val_sample_weight;

            Function val_f;
            List<Array> val_ins;

            // Prepare validation data.
            bool do_validation = false;
            if (validation_data != null)
            {
                do_validation = true;
                if (validation_data.Count == 2)
                {
                    val_x = validation_data[0];
                    val_y = validation_data[1];
                    val_sample_weight = null;
                }
                else if (validation_data.Count == 3)
                {
                    val_x = validation_data[0];
                    val_y = validation_data[1];
                    val_sample_weight = validation_data[2];
                }
                else
                {
                    throw new Exception($"When passing validation_data, it must contain 2 (x_val, y_val) or 3 (x_val, y_val, val_sample_weights) items, however it contains {validation_data.Count} items.");
                }

                var (val_xx, val_yy, val_sample_weightsx) = this._standardize_user_data(val_x, val_y, sample_weight: val_sample_weight, check_batch_axis: false, batch_size: batch_size);

                this._make_test_function();
                val_f = this.test_function;

                val_ins = new List<Array>();
                if (this.uses_learning_phase && !(K.learning_phase() is int))
                {
                    val_ins = (val_xx.Concat(val_yy).Concat(val_sample_weightsx).Concat(new List<Array>() { new[] { 0.0 } })).ToList();
                }
                else
                {
                    val_ins = val_xx.Concat(val_yy).Concat(val_sample_weightsx).ToList();
                }
            }
            else if (validation_split != 0 && (0.0 < validation_split) && (validation_split < 1.0))
            {
                do_validation = true;
                int split_at;
                if (xx[0].GetLength() != null)
                    split_at = (int)(xx[0].GetLength(0) * (1.0 - validation_split));
                else
                    split_at = (int)(xx[0].Length * (1.0 - validation_split));

                List<Array> val_xx, val_yy;
                (xx, val_xx) = (_slice_arrays(xx, new[] { 0 }, split_at), _slice_arrays(xx, new[] { split_at }));
                (yy, val_yy) = (_slice_arrays(yy, new[] { 0 }, split_at), _slice_arrays(yy, new[] { split_at }));

                List<Array> sample_weights = _slice_arrays(sample_weightsx, new[] { 0 }, split_at);
                List<Array> val_sample_weights = _slice_arrays(sample_weightsx, new[] { split_at });

                this._make_test_function();
                val_f = this.test_function;

                if (this.uses_learning_phase && !(K.learning_phase() is int))
                {
                    val_ins = val_xx.Concat(val_yy).Concat(val_sample_weights).Concat(new List<Array>() { new[] { 0.0 } }).ToList();
                }
                else
                {
                    val_ins = val_xx.Concat(val_yy).Concat(val_sample_weights).ToList();
                }
            }
            else
            {
                do_validation = false;
                val_f = null;
                val_ins = null;
            }

            List<Array> ins;

            // Prepare input arrays and training function.
            if (this.uses_learning_phase && !(K.learning_phase() is int))
            {
                ins = xx.Concat(yy).Concat(sample_weightsx).Concat(new List<Array>() { new[] { 1.0 } }).ToList();
            }
            else
            {
                ins = xx.Concat(yy).Concat(sample_weightsx).ToList();
            }

            this._make_train_function();
            var f = this.train_function;

            // Prepare display labels.
            var out_labels = this._get_deduped_metrics_names();
            List<String> callback_metrics;

            if (do_validation)
            {
                callback_metrics = (new List<string>(out_labels)).Concat(out_labels.Select(n => "val_" + n)).ToList();
            }
            else
            {
                callback_metrics = out_labels;
            }

            // Delegate logic to `_fit_loop`.
            return this._fit_loop(f, ins, out_labels: out_labels,
                                  batch_size: batch_size, epochs: epochs,
                                  verbose: verbose, callbacks: callbacks,
                                  val_f: val_f, val_ins: val_ins, shuffle: shuffle,
                                  callback_metrics: callback_metrics,
                                  initial_epoch: initial_epoch);
        }

        private List<Tensor> _slice_arrays(List<Tensor> sample_weightsx, int split_at)
        {
            throw new NotImplementedException();
        }

        private List<Tensor> _slice_arrays(List<Tensor> sample_weightsx, int v, int split_at)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Returns the loss value & metrics values for the model in test mode.
        /// </summary>
        /// <remarks>
        ///   Computation is done in batches.
        /// </remarks>
        /// <param name="x">The Numpy array of test data, or list of Numpy arrays if the model has multiple inputs.
        ///   If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">The Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. 
        ///   If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="batch_size">The number of samples per gradient update.</param>
        /// <param name="verbose">The verbosity mode, 0 or 1.</param>
        /// <param name="sample_weights">The array of weights to weight the contribution of different samples to 
        ///   the loss and metrics.</param>
        ///   
        /// <returns>
        ///   Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has 
        ///   multiple outputs and/or metrics). The attribute `model.metrics_names` will give you the display labels for 
        ///   the scalar outputs.
        /// </returns>
        /// 
        public Array evaluate(Array x, Array y, int batch_size = 32, int verbose = 1, Array sample_weight = null)
        {
            return evaluate(x.dict_from_single(), y.dict_from_single(), batch_size, verbose, sample_weight.dict_from_single())[0];
        }

        /// <summary>
        ///   Returns the loss value & metrics values for the model in test mode.
        /// </summary>
        /// <remarks>
        ///   Computation is done in batches.
        /// </remarks>
        /// <param name="x">The Numpy array of test data, or list of Numpy arrays if the model has multiple inputs.
        ///   If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">The Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. 
        ///   If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="batch_size">The number of samples per gradient update.</param>
        /// <param name="verbose">The verbosity mode, 0 or 1.</param>
        /// <param name="sample_weights">The array of weights to weight the contribution of different samples to 
        ///   the loss and metrics.</param>
        ///   
        /// <returns>
        ///   Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has 
        ///   multiple outputs and/or metrics). The attribute `model.metrics_names` will give you the display labels for 
        ///   the scalar outputs.
        /// </returns>
        /// 
        public virtual Array[] evaluate(Dictionary<string, Array> x, Dictionary<string, Array> y, int batch_size = 32, int verbose = 1, Dictionary<string, Array> sample_weight = null)
        {
            // Validate user data.
            var (xx, yy, sample_weightsx) = this._standardize_user_data(x, y, sample_weight: sample_weight, check_batch_axis: false, batch_size: batch_size);

            return (Array[])evaluate(xx, yy, batch_size, verbose, sample_weightsx);
        }

        public virtual Array evaluate(List<Array> x, List<Array> y, int batch_size = 32, int verbose = 1, List<Array> sample_weight = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L1546
            // Prepare inputs, delegate logic to `_test_loop`.
            var ins = new List<Array>();
            if (this.uses_learning_phase && !(K.learning_phase() is int))
                ins = x.Concat(y).Concat(sample_weight).Concat(new List<double[]> { new[] { 0.0 } }).ToList();
            else
                ins = x.Concat(y).Concat(sample_weight.ToArray()).ToList();

            this._make_test_function();
            var f = this.test_function;
            //return this._test_loop(f, ins, batch_size: batch_size, verbose: verbose);
            return null;
        }

        /// <summary>
        ///   Generates output predictions for the input samples.
        /// </summary>
        /// 
        /// <remarks>
        ///   Computation is done in batches.
        /// </remarks>
        /// 
        /// <param name="x">The input data, as a Numpy array (or list of Numpy arrays if the model has multiple outputs).</param>
        /// <param name="batch_size">The size of the batch.</param>
        /// <param name="verbose">The verbosity mode, 0 or 1.</param>
        /// 
        /// <returns>Numpy array(s) of predictions.</returns>
        /// 
        public Array[] predict(Array x, int batch_size = 32, int verbose = 0)
        {
            return predict(x.dict_from_single(), batch_size, verbose);
        }

        /// <summary>
        ///   Generates output predictions for the input samples.
        /// </summary>
        /// 
        /// <remarks>
        ///   Computation is done in batches.
        /// </remarks>
        /// 
        /// <param name="x">The input data, as a Numpy array (or list of Numpy arrays if the model has multiple outputs).</param>
        /// <param name="batch_size">The size of the batch.</param>
        /// <param name="verbose">The verbosity mode, 0 or 1.</param>
        /// 
        /// <returns>Numpy array(s) of predictions.</returns>
        /// 
        public virtual Array[] predict(Dictionary<string, Array> x, int batch_size = 32, int verbose = 0)
        {
            // Validate user data.
            List<Array> xx = _standardize_input_data(x, this._feed_input_names, this._feed_input_shapes, check_batch_axis: false);

            if (this.stateful)
            {
                var t = xx[0];
                if (t.GetLength(0) > batch_size && t.GetLength(0) % batch_size != 0)
                    throw new Exception($"In a stateful network, you should only pass inputs with a number of samples that can be divided by the batch size. Found: {t.GetLength(0)} samples. Batch size: {batch_size}.");
            }

            List<Array> ins;

            // Prepare inputs, delegate logic to `_predict_loop`.
            if (this.uses_learning_phase && !(K.learning_phase() is int))
                ins = xx.Concat(new List<double[]> { new[] { 0.0 } }).ToList();
            else
                ins = xx.ToList();

            this._make_predict_function();
            Function f = this.predict_function;
            return this._predict_loop(f, ins, batch_size: batch_size, verbose: verbose);
        }

        /// <summary>
        ///   Runs a single gradient update on a single batch of data.
        /// </summary>
        /// 
        /// <param name="x">The Numpy array of training data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">The Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="sample_weight">The optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param>
        /// <param name="class_weight">The optional dictionary mapping class indices (integers) to a weight (float) to apply to the model's loss for the samples from this class during training. This can be useful to tell the model to "pay more attention" to samples from an under-represented class.</param>
        /// 
        /// <returns>Scalar training loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute `model.metrics_names` will give you the display labels for the scalar outputs.</returns>
        /// 
        public virtual List<Tensor> train_on_batch(Dictionary<string, Array> x, Dictionary<string, Array> y, Dictionary<string, Array> sample_weight = null, Dictionary<string, Dictionary<string, double>> class_weight = null)
        {
            var (xx, yy, sample_weights) = this._standardize_user_data(x, y, sample_weight: sample_weight, class_weight: class_weight, check_batch_axis: true);

            return train_on_batch(xx, yy, sample_weights);
        }

        public virtual List<Tensor> train_on_batch(List<Array> x, List<Array> y, List<Array> sample_weightsx, Dictionary<string, Tensor> class_weight = null)
        {
            var ins = new List<Array>();
            if (this.uses_learning_phase && !(K.learning_phase() is int))
                ins = x.Concat(y).Concat(sample_weightsx).Concat(new List<Array>() { new[] { 1.0 } }).ToList();
            else
                ins = x.Concat(y).Concat(sample_weightsx).ToList();

            this._make_train_function();
            outputs = this.train_function.Call(ins);
            return outputs;
        }


        /// <summary>
        ///   Test the model on a single batch of samples.
        /// </summary>
        /// <param name="x">The Numpy array of test data, or list of Numpy arrays if the model has multiple inputs. If 
        ///   all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">The Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If 
        ///   all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="">The optional array of the same length as x, containing weights to apply to the model's loss for each sample.
        ///   In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight 
        ///   to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param>
        ///   
        /// <returns>
        ///   Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs
        //    and/or metrics). The attribute `model.metrics_names` will give you the display labels for the scalar outputs.
        /// </returns>
        ///   
        public virtual List<Tensor> test_on_batch(Dictionary<string, Array> x, Dictionary<string, Array> y, Dictionary<string, Array> sample_weight = null)
        {
            var (xx, yy, sample_weights) = this._standardize_user_data(x, y, sample_weight: sample_weight, check_batch_axis: true);

            return test_on_batch(xx, yy, sample_weights);
        }

        private List<Tensor> test_on_batch(List<Array> xx, List<Array> yy, List<Array> sample_weight)
        {
            var ins = new List<Array>();
            if (this.uses_learning_phase && !(K.learning_phase() is int))
            {
                ins = xx.Concat(yy).Concat(sample_weight).Concat(new List<Array>() { new[] { 0.0 } }).ToList();
            }
            else
            {
                ins = xx.Concat(yy).Concat(sample_weight).ToList();
            }

            this._make_test_function();
            outputs = this.test_function.Call(ins);
            return outputs;
        }

        public virtual List<Tensor> predict_on_batch(Dictionary<string, Array> x)
        {
            //"""Returns predictions for a single batch of samples.
            //// Arguments
            //    x: Input samples, as a Numpy array.
            //// Returns
            //    Numpy array(s) of predictions.
            //"""
            var xx = _standardize_input_data(x, this._feed_input_names, this._feed_input_shapes);

            return predict_on_batch(xx);
        }

        private List<Tensor> predict_on_batch(List<Array> x)
        {
            var ins = new List<Array>();

            if (this.uses_learning_phase && !(K.learning_phase() is int))
            {
                ins = x.Concat(new List<Array>() { new[] { 0.0 } }).ToList();
            }
            else
            {
                ins = x.ToList();
            }
            this._make_predict_function();
            outputs = this.predict_function.Call(ins);
            return outputs;
        }



        /// <summary>
        ///   Fits the model on data yielded batch-by-batch by a Python generator.
        /// </summary>
        /// <remarks>
        ///   The generator is run in parallel to the model, for efficiency. For instance, this allows you 
        ///   to do real-time data augmentation on images on CPU in parallel to training your model on GPU. 
        ///   The use of `keras.utils.Sequence` guarantees the ordering and guarantees the single use of every 
        ///   input per epoch when using `use_multiprocessing=true`.
        /// </remarks>
        /// 
        /// <param name="generator">The generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data 
        ///   when using multiprocessing. The output of the generator must be either: 1) a tuple (inputs, targets), 2) a tuple (inputs, targets, 
        ///   sample_weights). All arrays should contain the same number of samples. The generator is expected to loop over its data indefinitely. 
        ///   An epoch finishes when `steps_per_epoch` batches have been seen by the model.</param>
        /// <param name="steps_per_epoch">The Total number of steps (batches of samples) to yield from `generator` before declaring one epoch
        ///        finished and starting the next epoch. It should typically be equal to the number of unique samples if your dataset
        ///        divided by the batch size.</param>
        /// <param name="epochs">The total number of iterations on the data.</param>
        /// <param name="verbose">The verbosity mode, 0, 1, or 2.</param>
        /// <param name="callbacks">The list of callbacks to be called during training.</param>
        /// <param name="validation_data">The validation data. This can be either 1) a generator for the validation data,
        ///        2) a tuple (inputs, targets), 3) a tuple (inputs, targets, sample_weights).</param>
        /// <param name="validation_steps">The validation steps. Only relevant if `validation_data` is a generator. Total number of steps 
        ///        (batches of samples) to yield from `generator` before stopping.</param>
        /// <param name="class_weight">The class weight. A dictionary mapping class indices to a weight for the class.</param>
        /// <param name="max_queue_size">Maximum size of the queue for the generator queue.</param>
        /// <param name="workers">The maximum number of processes to spin up when using process based threading.</param>
        /// <param name="use_multiprocessing">Whether to use multiprocessing. If true, use process based threading.
        ///        Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the 
        ///        generator as they can't be passed easily to children processes.</param>
        /// <param name="initial_epoch">The initial epoch at which to start training
        ///        (useful for resuming a previous training run).</param>
        ///        
        /// <returns>A `History` object.</returns>
        /// 
        public History fit_generator(IEnumerator<List<Dictionary<string, Array>>> generator,
                                  int steps_per_epoch,
                                  int epochs = 1,
                                  int verbose = 1,
                                  CallbackList callbacks = null,
                                  IList<Dictionary<string, Array>> validation_data = null,
                                  int? validation_steps = null,
                                  Dictionary<int, Tensor> class_weight = null,
                                  int max_queue_size = 10,
                                  int workers = 1,
                                  bool use_multiprocessing = false,
                                  int initial_epoch = 0)
        {
            double wait_time = 0.01; // in seconds
            int epoch = initial_epoch;

            bool do_validation = validation_data != null;
            this._make_train_function();
            if (do_validation)
                this._make_test_function();

            // python 2 has 'next', 3 has '__next__'
            // avoid any explicit version checks
            bool val_gen = (validation_data is IList);

            if (val_gen && validation_steps != null)
                throw new Exception("When using a generator for validation data, you must specify a value for `validation_steps`.");

            // Prepare display labels.
            var out_labels = this._get_deduped_metrics_names();
            var callback_metrics = out_labels.Concat(out_labels.Select(n => "val_" + n));

            // prepare callbacks
            this.history = new History();
            CallbackList c = new CallbackList();
            c.Add(new BaseLogger());
            c.AddRange(callbacks);
            c.Add(history);
            if (verbose != 0)
                c.Add(new ProgbarLogger(count_mode: "steps"));
            callbacks = c;

            Model callback_model = null;

            // it's possible to callback a different model than this:
            if (this.callback_model != null)
            {
                callback_model = this.callback_model;
            }
            else
            {
                callback_model = this;
                callbacks.set_model(callback_model);
                callbacks.set_params(new Dictionary<string, object>
                {
                    { "epochs", epochs },
                    { "steps" ,  steps_per_epoch },
                    { "verbose" ,  verbose },
                    { "do_validation" ,  do_validation },
                    { "metrics" , callback_metrics },
                });
                callbacks.on_train_begin();
            }

            if (do_validation && !val_gen)
            {
                Dictionary<string, Array> val_x, val_y, val_sample_weight;
                if (validation_data.Count == 2)
                {
                    val_x = validation_data[0];
                    val_y = validation_data[1];
                    val_sample_weight = null;
                }
                else if (validation_data.Count == 3)
                {
                    val_x = validation_data[0];
                    val_y = validation_data[1];
                    val_sample_weight = validation_data[2];
                }
                else
                {
                    throw new Exception($"`validation_data` should be a tuple `(val_x, val_y, val_sample_weight)` or `(val_x, val_y)`. Found: {validation_data}.");
                }

                var (val_xx, val_yy, val_sample_weights) = this._standardize_user_data(val_x, val_y, val_sample_weight);

                List<Array> val_data = val_xx.Concat(val_yy).Concat(val_sample_weights).ToList();

                if (this.uses_learning_phase && !(K.learning_phase() is int))
                {
                    val_data.Add(new[] { 0.0 });

                    bool is_sequence = false;
                    foreach (var cbk in callbacks)
                    {
                        cbk.validation_data = val_data;
                        is_sequence = generator is Sequence;
                    }

                    if (!is_sequence && use_multiprocessing)
                    {
                        Trace.TraceWarning("Using a generator with `use_multiprocessing=true`"
                                        + " may duplicate your data.Please consider using"
                                        + " the `keras.utils.Sequence` class.");
                    }

                    Enqueuer enqueuer = null;

                    try
                    {
                        if (is_sequence)
                        {
                            enqueuer = new OrderedEnqueuer(generator,
                                                       use_multiprocessing: use_multiprocessing);
                        }
                        else
                        {
                            enqueuer = new GeneratorEnqueuer(generator,
                                                         use_multiprocessing: use_multiprocessing,
                                                         wait_time: wait_time);
                        }

                        enqueuer.start(workers: workers, max_queue_size: max_queue_size);
                        var output_generator = enqueuer.GetEnumerator();

                        callback_model.stop_training = false;

                        while (epoch < epochs)
                        {
                            callbacks.on_epoch_begin(epoch);
                            int steps_done = 0;
                            int batch_index = 0;

                            while (steps_done < steps_per_epoch)
                            {
                                List<List<Tensor>> generator_output = output_generator.Current;
                                output_generator.MoveNext();

                                List<Tensor> x = generator_output[0];
                                List<Tensor> y = generator_output[1];
                                List<Tensor> sample_weight = null;

                                if (generator_output.Count == 3)
                                    sample_weight = generator_output[2];

                                // build batch logs
                                var batch_logs = new Dictionary<string, object>();

                                int batch_size;
                                if (x is IList)
                                {
                                    batch_size = x[0].shape[0].Value;
                                }
                                else if (x is IDictionary)
                                {
                                    batch_size = x[0].shape[0].Value;
                                }
                                else
                                {
                                    batch_size = x[0].shape[0].Value;
                                }


                                batch_logs["batch"] = batch_index;
                                batch_logs["size"] = batch_size;

                                callbacks.on_batch_begin(batch_index, batch_logs);

                                List<Tensor> outs = this.train_on_batch(x, y,
                                                           sample_weight: sample_weight,
                                                           class_weight: class_weight);

                                foreach (var (l, o) in Enumerable.Zip(out_labels, outs, (a, b) => Tuple.Create(a, b)))
                                {
                                    batch_logs[l] = o;
                                }

                                callbacks.on_batch_end(batch_index, batch_logs);

                                // Construct epoch logs.
                                var epoch_logs = new Dictionary<string, object>();
                                batch_index += 1;
                                steps_done += 1;

                                // Epoch finished.
                                if (steps_done >= steps_per_epoch && do_validation)
                                {
                                    object val_outs;

                                    if (val_gen)
                                    {
                                        val_outs = this.evaluate_generator(
                                            validation_data,
                                            validation_steps.Value,
                                            max_queue_size: max_queue_size,
                                            workers: workers,
                                            use_multiprocessing: use_multiprocessing);
                                    }
                                    else
                                    {
                                        // No need for try/except because
                                        // data has already been validated.
                                        val_outs = this.evaluate(
                                            val_x, val_y,
                                            batch_size: batch_size,
                                            sample_weight: val_sample_weight,
                                            verbose: 0);
                                    }

                                    // Same labels assumed.
                                    //foreach (var (l, o) in Enumerable.Zip(out_labels, val_outs, (a, b) => Tuple.Create(a, b)))
                                    //{
                                    //    epoch_logs["val_" + l] = o;
                                    //}
                                }

                                callbacks.on_epoch_end(epoch, epoch_logs);
                                epoch += 1;
                                if (callback_model.stop_training)
                                    break;
                            }
                        }
                    }
                    finally
                    {
                        if (enqueuer != null)
                            enqueuer.stop();
                    }
                }
            }

            callbacks.on_train_end();
            return this.history;
        }



        public List<Tensor> evaluate_generator(object generator, int steps,
                               int max_queue_size = 10,
                               int workers = 1,
                               bool use_multiprocessing = false)
        {
            throw new NotImplementedException();
            /*
            this._make_test_function();

            int steps_done = 0;
            double wait_time = 0.01;
            List<Tensor> all_outs = new List<Tensor>();
            List<int> batch_sizes = new List<int>();
            bool is_sequence = generator is Sequence;

            if (!is_sequence && use_multiprocessing)
                Trace.TraceWarning("Using a generator with `use_multiprocessing=true` may duplicate your data. Please consider using the `keras.utils.Sequence` class.");

            Enqueuer enqueuer = null;

            try
            {
                if (is_sequence)
                    enqueuer = new OrderedEnqueuer(generator, use_multiprocessing: use_multiprocessing);
                else
                    enqueuer = new GeneratorEnqueuer(generator, use_multiprocessing: use_multiprocessing, wait_time: wait_time);

                enqueuer.start(workers: workers, max_queue_size: max_queue_size);
                IEnumerator<List<List<Tensor>>> output_generator = enqueuer.GetEnumerator();

                List<List<Tensor>> generator_output = null;
                while (steps_done < steps)
                {
                    generator_output = output_generator.Current;
                    output_generator.MoveNext();
                }

                List<Tensor> x, y;
                List<Tensor> sample_weight;
                if (generator_output.Count == 2)
                {
                    x = generator_output[0];
                    y = generator_output[1];
                    sample_weight = null;
                }
                else if (generator_output.Count == 3)
                {
                    x = generator_output[0];
                    y = generator_output[1];
                    sample_weight = generator_output[2];
                }
                else
                {
                    throw new Exception($"Output of generator should be a tuple (x, y, sample_weight) or (x, y). Found: {generator_output}.");
                }

                List<Tensor> outs = this.test_on_batch(x, y, sample_weight: sample_weight);

                int batch_size;

                if (x is IList<Tensor>)
                {
                    batch_size = x[0].shape[0].Value;
                }
                else if (x is IDictionary)
                {
                    batch_size = x[0].shape[0].Value;
                }
                else
                {
                    batch_size = x[0].shape[0].Value;
                }

                if (batch_size == 0)
                {
                    throw new Exception("Received an empty batch. Batches should at least contain one item.");
                }

                all_outs.AddRange(outs);

                steps_done += 1;
                batch_sizes.Add(batch_size);
            }
            finally
            {
                if (enqueuer != null)
                    enqueuer.stop();
            }


            return weightedMean(all_outs, batch_sizes);

            //var averages = new List<object>();
            //foreach (int i in Vector.Range(outs))
            //    averages.Add(weightedMean(all_outs.Select(o => o[i]).ToList(), batch_sizes: batch_sizes));

            //return averages;
            */
        }

        private static List<Tensor> weightedMean(List<Tensor> all_outs, List<int> batch_sizes)
        {
            throw new NotImplementedException();
        }

        public List<List<Tensor>> predict_generator(IEnumerator<List<Dictionary<string, Array>>> generator, int steps,
                              int max_queue_size = 10,
                              int workers = 1,
                              bool use_multiprocessing = false,
                              int verbose = 0)
        {
            //"""Generates predictions for the input samples from a data generator.
            //            The generator should return the same kind of data as accepted by
            //            `predict_on_batch`.
            //// Arguments
            //    generator: Generator yielding batches of input samples
            //                        or an instance of Sequence(keras.utils.Sequence)
            //                        object in order to avoid duplicate data
            //            when using multiprocessing.
            //    steps: Total number of steps(batches of samples)
            //                to yield from `generator` before stopping.
            //    max_queue_size: Maximum size for the generator queue.
            //    workers: Maximum number of processes to spin up
            //        when using process based threading
            //    use_multiprocessing: If `true`, use process based threading.
            //        Note that because
            //                this implementation relies on multiprocessing,
            //        you should not pass
            //        non picklable arguments to the generator
            //        as they can't be passed
            //                easily to children processes.
            //    verbose: verbosity mode, 0 or 1.
            //        // Returns
            //    Numpy array(s) of predictions.
            //// Raises
            //    ValueError: In case the generator yields
            //        data in an invalid format.
            //        """
            this._make_predict_function();

            int steps_done = 0;
            double wait_time = 0.01;
            List<List<Tensor>> all_outs = new List<List<Tensor>>();
            bool is_sequence = generator is Sequence;
            if (!is_sequence && use_multiprocessing)
            {
                Trace.TraceWarning("Using a generator with `use_multiprocessing=true` may duplicate your data. Please consider using the `keras.utils.Sequence` class.");
            }

            Enqueuer enqueuer = null;

            try
            {
                if (is_sequence)
                {
                    enqueuer = new OrderedEnqueuer(generator, use_multiprocessing: use_multiprocessing);
                }
                else
                {
                    enqueuer = new GeneratorEnqueuer(generator, use_multiprocessing: use_multiprocessing, wait_time: wait_time);
                    enqueuer.start(workers: workers, max_queue_size: max_queue_size, output_generator: enqueuer.GetEnumerator());
                }


                Progbar progbar = null;
                if (verbose == 1)
                    progbar = new Progbar(target: steps);

                while (steps_done < steps)
                {
                    List<Dictionary<string, Array>> generator_output = generator.Current;
                    generator.MoveNext();

                    Dictionary<string, Array> x;

                    // Compatibility with the generators
                    // used for training.
                    if (generator_output.Count == 2)
                    {
                        x = generator_output[0];
                    }
                    else if (generator_output.Count == 3)
                    {
                        x = generator_output[0];
                    }
                    else
                    {
                        // Assumes a generator that only
                        // yields inputs (not targets and sample weights).
                        x = generator_output[0];
                    }

                    // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/training.py#L2124
                    var outs = this.predict_on_batch(x);

                    if (all_outs.Count == 0)
                    {
                        foreach (var o in outs)
                        {
                            all_outs.Add(new List<Tensor>());

                            for (int i = 0; i < outs.Count; i++)
                            {
                                all_outs[i].Add(o);
                                steps_done += 1;
                            }
                        }
                    }

                    if (verbose == 1)
                        progbar.update(steps_done);
                }

                return all_outs;
            }
            finally
            {
                if (enqueuer != null)
                    enqueuer.stop();
            }
        }

        private ValueTuple<List<Tensor>, List<Tensor>, List<Tensor>> _standardize_user_data(List<Tensor> val_x, List<Tensor> val_y, List<Tensor> val_sample_weight)
        {
            throw new NotImplementedException();
        }

        private ValueTuple<List<Tensor>, List<Tensor>, List<Tensor>> _standardize_user_data(List<Tensor> val_x, List<Tensor> val_y, List<Tensor> val_sample_weight, IEnumerable<Tensor> val_data)
        {
            throw new NotImplementedException();
        }

        private List<Tensor> train_on_batch(List<Tensor> x, List<Tensor> y, List<Tensor> sample_weight, object class_weight)
        {
            throw new NotImplementedException();
        }

        private List<Tensor> test_on_batch(Tensor x, Tensor y, Tensor sample_weight)
        {
            throw new NotImplementedException();
        }
    }
}
