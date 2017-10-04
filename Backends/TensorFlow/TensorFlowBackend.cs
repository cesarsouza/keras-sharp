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

namespace KerasSharp.Backends
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Losses;
    using KerasSharp.Models;
    using TensorFlow;
    using static KerasSharp.Python;
    using Accord.Math;

    // TODO:

    public class TensorFlowBackend : BackendBase, IBackend
    {
        internal TFGraph tf;


        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L25

        // This is the default internal TF session used by Keras.
        // It can be set manually via `set_session(sess)`.
        internal TFSession _SESSION;

        // This dictionary holds a mapping {graph: learning_phase}.
        // A learning phase is a bool tensor used to run Keras models in
        // either train mode (learning_phase == 1) or test mode (learning_phase == 0).
        private Dictionary<TFGraph, TFOutput> _GRAPH_LEARNING_PHASES = new Dictionary<TFGraph, TFOutput>();

        // This dictionary holds a mapping {graph: UID_DICT}.
        // each UID_DICT is a dictionary mapping name prefixes to a current index,
        // used for generatic graph-specific string UIDs
        // for various names (e.g. layer names).
        private Dictionary<TFGraph, Dictionary<string, int>> _GRAPH_UID_DICTS = new Dictionary<TFGraph, Dictionary<string, int>>();

        // This boolean flag can be set to True to leave variable initialization
        // up to the user.
        // Change its value via `manual_variable_initialization(value)`.
        bool _MANUAL_VAR_INIT = false;


        public TensorFlowBackend()
        {
            this.tf = new TFGraph();
            this._SESSION = new TFSession(tf);
        }

        /// <summary>
        ///   Get the uid for the default graph.
        /// </summary>
        /// 
        /// <param name="prefix">An optional prefix of the graph.</param>
        /// 
        /// <returns>A unique identifier for the graph.</returns>
        /// 
        public int get_uid(string prefix)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L58
            var graph = tf;
            if (!_GRAPH_UID_DICTS.ContainsKey(graph))
                _GRAPH_UID_DICTS[graph] = new Dictionary<string, int>();
            if (!_GRAPH_UID_DICTS[graph].ContainsKey(prefix))
                _GRAPH_UID_DICTS[graph][prefix] = 0;
            _GRAPH_UID_DICTS[graph][prefix] += 1;
            return _GRAPH_UID_DICTS[graph][prefix];
        }

        /// <summary>
        ///   Reset graph identifiers.
        /// </summary>
        /// 
        public void reset_uids()
        {
            _GRAPH_UID_DICTS = new Dictionary<TFGraph, Dictionary<string, int>>();
        }

        /// <summary>
        ///   Destroys the current TF graph and creates a new one.
        ///   Useful to avoid clutter from old models / layers.
        /// </summary>
        /// 
        public void clear_session()
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L71
            // tf.reset_default_graph();
            tf = new TFGraph();
            _SESSION = new TFSession(tf);
            //
            reset_uids();
            TFOutput phase = tf.Placeholder(dtype: TFDataType.Bool, operName: "keras_learning_phase");
            _GRAPH_LEARNING_PHASES = new Dictionary<TFGraph, TFOutput>();
            _GRAPH_LEARNING_PHASES[tf] = phase;
        }





        /// <summary>
        ///   Reshapes a tensor to the specified shape.
        /// </summary>
        /// 
        /// <param name="x">The Tensor or variable.</param>
        /// <param name="shape">The target shape.</param>
        /// 
        /// <returns>Tensor.</returns>
        /// 
        public Tensor reshape(Tensor x, int[] shape)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L1724
            return Out(tf.Reshape(tensor: In(x), shape: _constant(shape)));
        }



        public Tensor abs(Tensor input)
        {
            return Out(tf.Abs(In(input)));
        }






        public List<Array> batch_get_value(List<Tensor> weights)
        {
            throw new NotImplementedException();
        }

        public List<Array> batch_get_value(List<List<Tensor>> weights)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<Tuple<Tensor, Array>> weight_value_tuples)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<(Tensor, Array)> tuples)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Binary crossentropy between an output tensor and a target tensor.
        /// </summary>
        /// 
        /// <param name="output">A tensor.</param>
        /// <param name="target">A tensor of the same shape as `output`.</param>
        /// <param name="from_logits">Whether `output` is expected to be a logits tensor. By default, we consider that `output` encodes a probability distribution.</param>
        /// 
        /// <returns>Output tensor.</returns>
        /// 
        public Tensor binary_crossentropy(Tensor output, Tensor target, bool from_logits = false)
        {
            var _output = In(output);
            var _target = In(target);

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2792
            // Note: tf.nn.sigmoid_cross_entropy_with_logits
            // expects logits, Keras expects probabilities.
            if (!from_logits)
            {
                // transform back to logits
                TFOutput _epsilon = _constant(epsilon(), dtype: _output.dtype);
                TFOutput o = _output.output;
                o = tf.ClipByValue(o, _epsilon, tf.Sub(_constant(1f), _epsilon));
                o = tf.Log(tf.Div(_output, (tf.Sub(tf.Const(1f), _output))));
            }

            return Out(tf.sigmoid_cross_entropy_with_logits(labels: _target, logits: _output));
        }

        public Tensor cast(Tensor x, DataType dataType)
        {
            return Out(tf.Cast(In(x), In(dataType)));
        }

        /// <summary>
        ///   Categorical crossentropy between an output tensor and a target tensor.
        /// </summary>
        /// 
        /// <param name="target">A tensor of the same shape as `output`.</param>
        /// <param name="output">A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected to be the logits).</param>
        /// <param name="from_logits">Boolean, whether `output` is the result of a softmax, or is a tensor of logits.</param>
        /// 
        /// <returns>Output tensor.</returns>
        /// 
        public Tensor categorical_crossentropy(Tensor target, Tensor output, bool from_logits = false)
        {
            var _target = In(target);
            var _output = In(output);

            // Note: tf.nn.softmax_cross_entropy_with_logits
            // expects logits, Keras expects probabilities.
            if (!from_logits)
            {
                // scale preds so that the class probas of each sample sum to 1
                int?[] shape = output.shape;
                var last = tf.Const(new TFTensor(shape.Length - 1));
                TFOutput o = tf.Div(_output, tf.ReduceSum(_output, axis: last, keep_dims: true));
                // manual computation of crossentropy
                TFOutput _epsilon = _constant(epsilon(), dtype: _output.dtype);
                o = tf.ClipByValue(o, _epsilon, tf.Sub(_constant(1f), _epsilon));
                o = tf.Neg(tf.ReduceSum(tf.Mul(_target, tf.Log(_output)), axis: last));
                return Out(o);
            }

            return Out(tf.SoftmaxCrossEntropyWithLogits(_target, _output).loss);
        }

        public Tensor clip(Tensor norms, int v, int maxValue)
        {
            throw new NotImplementedException();
        }

        public Tensor clip(Tensor norms, double min_value, double max_value)
        {
            throw new NotImplementedException();
        }

        public Tensor clip_norm(Tensor g, double clipnorm, Tensor norm)
        {
            throw new NotImplementedException();
        }

        public Tensor constant<T>(T value, int?[] shape = null, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            int[] _shape;
            if (shape == null)
            {
                Array arr = value as Array;
                if (arr != null)
                    _shape = arr.GetLength();
                else _shape = new int[0];
                shape = _shape.Select(x => (int?)x).ToArray();
            }
            else
            {
                _shape = shape.Select(x => x.Value).ToArray();
            }

            TFOutput o;
            if (shape != null && !(value is Array))
            {
                o = _constant(MatrixEx.Create(value.GetType(), _shape, value), In(dtype.Value), name);
            }
            else
            {
                o = _constant(value, In(dtype.Value), name);
            }

            if (!_int_shape(o).IsEqual(shape))
                throw new Exception();

            return Out(o);
        }

        private TFOutput _constant<T>(T value, TFDataType? dtype = null, string name = null)
        {
            TFTensor t = new TFTensor((dynamic)value);

            TFOutput o = tf.Const(t, operName: name);

            if (dtype == null || o.OutputType == dtype.Value)
                return o;

            return tf.Cast(o, dtype.Value);
        }




        public Tensor dropout(object p, double retain_prob, object noise_shape, object seed)
        {
            throw new NotImplementedException();
        }

        public DataType? dtype(Tensor tensor)
        {
            return Out(In(tensor).dtype);
        }

        public Tensor elu(Tensor x)
        {
            return Out(tf.Elu(In(x)));
        }

        public Tensor elu(Tensor x, double alpha)
        {
            throw new NotImplementedException();
        }

        public Tensor exp(Tensor x)
        {
            return Out(tf.Exp(In(x)));
        }

        public Function function(List<Tensor> inputs, List<Tensor> outputs, List<List<Tensor>> updates, string name)
        {
            return new TFFunction(this, inputs: inputs, outputs: outputs, updates: updates, name: name);
        }



        /// <summary>
        ///   Returns the shape of a variable.
        /// </summary>
        /// 
        public int?[] get_variable_shape(Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2192
            return int_shape(x);
        }

        public List<Tensor> gradients(Tensor loss, object param)
        {
            throw new NotImplementedException();
        }

        public List<Tensor> gradients(Tensor loss, List<Tensor> param)
        {
            var y = new TFOutput[] { In(loss).output };
            var x = param.Select(t => In(t).output).ToArray();
            TFOutput[] grads = tf.AddGradients(x, y);
            List<Tensor> r = grads.Select(o => Out(o)).ToList();
            return r;
        }

        public Tensor greater_equal(Tensor w, double v)
        {
            throw new NotImplementedException();
        }

        public Tensor not_equal(Tensor x, Tensor y)
        {
            return Out(tf.NotEqual(In(x), In(y)));
        }

        public Tensor not_equal(Tensor x, double y)
        {
            return Out(tf.NotEqual(In(x), tf.Const(y, In(x).dtype)));
        }

        public Tensor hard_sigmoid(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor identity(Tensor x)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Returns the shape tensor or variable as a tuple of int or None entries.
        /// </summary>
        /// 
        /// <param name="x">Tensor or variable.</param>
        /// 
        /// <returns>A tuple of integers(or None entries).</returns>
        /// 
        public int?[] int_shape(Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L468

            if (x._keras_shape != null)
                return x._keras_shape;

            return _int_shape(In(x).output);
        }

        private int?[] _int_shape(TFOutput _x)
        {
            try
            {
                long[] shape = tf.GetTensorShape(_x).ToArray();
                return shape.Select(i => i == -1 ? null : (int?)i).ToArray();
            }
            catch
            {
                return null;
            }
        }

        public int?[] int_shape(TFTensor input_tensor)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Selects `x` in train phase, and `alt` otherwise.
        /// </summary>
        /// 
        /// <param name="x">What to return in train phase.</param>
        /// <param name="alt">What to return otherwise.</param>
        /// <param name="training">Optional scalar tensor specifying the learning phase.</param>
        /// 
        /// <returns>Either 'x' or 'alt' based on the 'training' flag. The 'training' flag defaults to 'K.learning_phase()'.</returns>
        /// 
        public Tensor in_train_phase(Func<Tensor> x, Func<Tensor> alt, bool? training)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2583

            bool uses_learning_phase;

            if (training == null)
            {
                training = (bool)learning_phase();
                uses_learning_phase = true;
            }
            else
            {
                uses_learning_phase = false;
            }

            if (training == true)
            {
                return x();
            }
            else if (training == false)
            {
                return alt();
            }
            else
            {
                //else: assume learning phase is a placeholder tensor.
                throw new NotImplementedException();
            }

            // Tensor xx = @switch(training, x, alt);

            if (uses_learning_phase)
                x()._uses_learning_phase = true;
            return x();
        }

        /// <summary>
        ///   Switches between two operations depending on a scalar value. Note that both `then_expression` and `else_expression`
        ///   should be symbolic tensors of the *same shape
        /// </summary>
        /// 
        /// <param name="condition">The condition: scalar tensor(`int` or `bool`).</param>
        /// <param name="then_expression">Either a tensor, or a callable that returns a tensor.</param>
        /// <param name="else_expression">Either a tensor, or a callable that returns a tensor.</param>
        /// 
        /// <returns>The selected tensor.</returns>
        /// 
        public Tensor @switch(Tensor condition, Func<Tensor> then_expression, Func<Tensor> else_expression)
        {
            var _condition = In(condition);

            if (_condition.dtype != TFDataType.Bool)
                condition = Out(tf.Cast(_condition, TFDataType.Bool));

            throw new NotImplementedException();

            //TFOutput x = tf.cond(condition,
            //            () => then_expression().output,
            //            () => else_expression().output);
            //return tensor(x);
        }

        public bool is_sparse(Tensor tensor)
        {
            return false;
        }

        public Tensor l2_normalize(Tensor expected, int axis)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Returns the learning phase flag.
        /// </summary>
        /// 
        /// <remarks>
        ///   The learning phase flag is a bool tensor(0 = test, 1 = train)
        ///   to be passed as input to any Keras function
        ///   that uses a different behavior at train time and test time.
        /// </remarks>
        /// 
        /// <returns> Learning phase (scalar integer tensor or Python integer).</returns>
        /// 
        public object learning_phase()
        {
            TFGraph graph = tf;
            if (!_GRAPH_LEARNING_PHASES.ContainsKey(graph))
            {
                TFOutput phase = tf.Placeholder(dtype: TFDataType.Bool, operName: "keras_learning_phase");
                _GRAPH_LEARNING_PHASES[graph] = phase;
            }

            return Out(_GRAPH_LEARNING_PHASES[graph]);
        }

        public Tensor max(Tensor x, int v, object p)
        {
            throw new NotImplementedException();
        }

        public Tensor max(Tensor x, int axis, bool keepdims)
        {
            throw new NotImplementedException();
        }

        public Tensor max(Tensor tensor, int axis)
        {
            throw new NotImplementedException();
        }

        public Tensor maximum(double v, Tensor tensor)
        {
            throw new NotImplementedException();
        }



        public TFOutput _normalize_axis(int[] axis, int? ndim)
        {
            axis = (int[])axis.Clone();
            for (int i = 0; i < axis.Length; i++)
            {
                if (axis[i] < 0)
                    axis[i] = axis[i] % ndim.Value;
            }

            return tf.Const(axis);
        }

        /// <summary>
        ///   Mean of a tensor, alongside the specified axis.
        /// </summary>
        /// 
        /// <param name="x">A tensor or variable.</param>
        /// <param name="axis">A list of integer. Axes to compute the mean.</param>
        /// <param name="keepdims>A boolean, whether to keep the dimensions or not. If <paramref name="keepdims"/> is <c>false</c>, 
        ///   the rank of the tensor is reduced by 1 for each entry in <paramref name="axis"/>. If <paramref name="keepdims"/> is 
        ///   <c>true</c>, the reduced dimensions are retained with length 1.
        ///   
        /// <returns>A tensor with the mean of elements of <c>x</c>.</returns>
        /// 
        public Tensor mean(Tensor x, int[] axis, bool keepdims = false, string name = null)
        {
            return Out(tf.ReduceMean(In(x), _normalize_axis(axis, ndim(x)), keepdims, operName: name));
        }

        /// <summary>
        ///   Mean of a tensor, alongside the specified axis.
        /// </summary>
        /// 
        /// <param name="x">A tensor or variable.</param>
        /// <param name="axis">The axis where to compute the mean.</param>
        /// <param name="keepdims>A boolean, whether to keep the dimensions or not. If <paramref name="keepdims"/> is <c>false</c>, 
        ///   the rank of the tensor is reduced by 1 for each entry in <paramref name="axis"/>. If <paramref name="keepdims"/> is 
        ///   <c>true</c>, the reduced dimensions are retained with length 1.
        ///   
        /// <returns>A tensor with the mean of elements of <c>x</c>.</returns>
        /// 
        public Tensor mean(Tensor x, int axis = -1, bool keepdims = false, string name = null)
        {
            return Out(tf.ReduceMean(In(x), axis: tf.Const(axis), keep_dims: keepdims, operName: name));
        }


        public Tensor minus(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor dot(Tensor a, Tensor b)
        {
            return Out(tf.MatMul(In(a).output, In(b).output));
        }


        public Tensor mul<T>(T a, Tensor b)
        {
            return mul(constant(a, dtype: dtype(b)), b);
        }

        public Tensor mul(Tensor a, Tensor b)
        {
            return Out(tf.Mul(In(a).output, In(b).output));
        }

        public Tensor mul<T>(Tensor a, T b)
        {
            return mul(a, constant(b, dtype: dtype(a)));
        }

        public Tensor mul(List<Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }





        public Tensor div<T>(T a, Tensor b)
        {
            return div(constant(a, dtype: dtype(b)), b);
        }

        public Tensor div(Tensor a, Tensor b)
        {
            return Out(tf.Mul(In(a).output, In(b).output));
        }

        public Tensor div<T>(Tensor a, T b)
        {
            return div(a, constant(b, dtype: dtype(a)));
        }

        public Tensor div(List<Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }



        public Tensor add(Tensor a, Tensor b)
        {
            return Out(tf.Add(In(a).output, In(b).output));
        }

        public Tensor bias_add(Tensor a, Tensor b)
        {
            return add(a, b);
        }

        public Tensor add<T>(T a, Tensor b)
        {
            return add(constant(a), b);
        }

        public Tensor add<T>(Tensor a, T b)
        {
            return add(a, constant(b));
        }



        public Tensor subtract(Tensor a, Tensor b)
        {
            return Out(tf.Sub(In(a).output, In(b).output));
        }

        public Tensor subtract<T>(T a, Tensor b)
        {
            return subtract(constant(a), b);
        }

        public Tensor subtract<T>(Tensor a, T b)
        {
            return subtract(a, constant(b));
        }



        public IDisposable name_scope(string name)
        {
            return tf.WithScope(name);
        }



        /// <summary>
        /// Returns the number of axes in a tensor, as an integer.
        /// </summary>
        /// <param name="x">Tensor or variable.</param>
        /// <example>
        /// <codesrc="TensorFlowBackendTest.cs" region="doc_ndim">
        /// </example>
        /// 
        public int? ndim(Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L519

            int?[] dims = x.shape;

            if (dims != null)
                return dims.Length;

            return tf.GetTensorNumDims(In(x).output);
        }

        public Tensor placeholder(int?[] shape = null, int? ndim = null, DataType? dtype = null, bool sparse = false, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L397

            if (sparse)
                throw new NotImplementedException();

            if (dtype == null)
                dtype = floatx();

            if (shape == null)
            {
                if (ndim != null)
                    shape = new int?[ndim.Value];
            }

            var tfshape = this.In(shape);

            Tensor x = Out(tf.Placeholder(In(dtype.Value), tfshape, operName: name));
            x._keras_shape = shape;
            x._uses_learning_phase = false;
            return x;
        }

        /// <summary>
        ///   Returns a tensor with uniform distribution of values.
        /// </summary>
        /// <param name="shape">A tuple of integers, the shape of tensor to create.</param>
        /// <param name="minval">A float, lower boundary of the uniform distribution to draw samples.</param>
        /// <param name="maxval">A float, upper boundary of the uniform distribution to draw samples.</param>
        /// <param name="dtype">The dtype of returned tensor.</param>
        /// <param name="seed">The random seed.</param>
        /// 
        /// <returns>A tensor.</returns>
        /// 
        public Tensor random_uniform(int?[] shape, double minval = 0.0, double maxval = 1.0, DataType? dtype = null, int? seed = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            var _dtype = In(dtype.Value);

            if (seed == null)
                seed = Accord.Math.Random.Generator.Random.Next(1_000_000);

            using (var scope = name_scope("random_uniform"))
            {
                var _shape = tf.Const(shape.Select(x => (long)x).ToArray());
                TFOutput u = tf.RandomUniform(_shape, dtype: _dtype, seed: seed, operName: name);

                return Out(tf.Add(tf.Mul(u, _constant(maxval - minval, dtype: _dtype)),
                                            _constant(minval, dtype: _dtype)));
            }
        }

        public Tensor relu(Tensor x)
        {
            return Out(tf.Relu(In(x)));
        }

        public Tensor sigmoid(Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2817
            return Out(tf.Sigmoid(In(x)));
        }

        public Tensor softmax(Tensor x)
        {
            return Out(tf.Softmax(In(x).output));
        }

        public Tensor softplus(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor softsign(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor sqrt(Tensor x)
        {
            return Out(tf.Sqrt(In(x)));
        }

        public Tensor square(Tensor w)
        {
            return Out(tf.Square(In(w)));
        }

        public Tensor sum(Tensor x, int[] axis, bool keepdims = false, string name = null)
        {
            return Out(tf.ReduceSum(In(x), tf.Const(axis), keepdims, name));
        }

        public Tensor sum(Tensor x, int axis, bool keepdims = false, string name = null)
        {
            return Out(tf.ReduceSum(In(x), tf.Const(axis), keepdims, name));
        }

        public Tensor sum(List<Tensor> x, int[] axis = null, bool keepdims = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public object sum(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor sum(double v, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor tanh(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor truncated_normal(int[] shape, double v, double stddev, DataType? dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor truncated_normal(int?[] shape, double v, double stddev, DataType? dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public List<Tensor> update(Tensor x, Tensor new_x)
        {
            return new List<Tensor>() { Out(tf.Assign(In(x), In(new_x))) };
        }

        public List<Tensor> update_add(Tensor iterations, int v)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Instantiates a variable and returns it.
        /// </summary>
        /// 
        /// <param name="value">C# array, initial value of the tensor.</param>
        /// <param name="dtype">Tensor type.</param>
        /// <param name="name">Optional name string for the tensor.</param>
        /// 
        public Tensor variable<T>(T value, string name = null)
            where T : struct
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308

            var t = new TensorFlowTensor(this);

            // trick for being type safe and still allow all numeric types supported by TFTensor 
            t.output = tf.Const(new TFTensor((dynamic)value), operName: name);
            t._keras_shape = new int?[] { };
            t._uses_learning_phase = false;
            return t;
        }

        /// <summary>
        ///   Instantiates a variable and returns it.
        /// </summary>
        /// 
        /// <param name="array">C# array, initial value of the tensor.</param>
        /// <param name="dtype">Tensor type.</param>
        /// <param name="name">Optional name string for the tensor.</param>
        /// 
        public Tensor variable(Array array, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308

            var t = new TensorFlowTensor(this);

            t.output = tf.Const(new TFTensor(array), operName: name);
            t._keras_shape = array.GetLength().Apply(x => (int?)x);
            t._uses_learning_phase = false;
            return t;
        }

        /// <summary>
        ///   Instantiates a variable and returns it.
        /// </summary>
        /// 
        /// <param name="tensor">Tensor, initial value of the tensor.</param>
        /// <param name="dtype">Tensor type.</param>
        /// <param name="name">Optional name string for the tensor.</param>
        /// 
        public Tensor variable(Tensor tensor, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            var _tensor = In(tensor);

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308

            var t = new TensorFlowTensor(this);

            if (_tensor.tensor == null)
                t.output = _tensor.output;
            else
                t.output = tf.Const(_tensor.tensor);
            t._keras_shape = tensor.shape;
            t._uses_learning_phase = false;
            return t;
        }

        public object eval(Tensor tensor)
        {
            var _tensor = In(tensor);

            TFTensor[] result = _SESSION.Run(new TFOutput[] { }, new TFTensor[] { }, new[] { _tensor.output });

            if (result.Length == 1)
                return result[0].GetValue();

            return result.Apply(x => x.GetValue());
        }




        /// <summary>
        ///   Instantiates an all-zeros variable and returns it.
        /// </summary>
        /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
        /// <param name="dtype">Data type of returned Keras variable.</param>
        /// <param name="name">String, name of returned Keras variable.</param>
        /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
        public Tensor zeros(int?[] shape, DataType? dtype = null, string name = null)
        {
            return zeros(shape.Select(i => i.Value).ToArray(), dtype, name);
        }

        /// <summary>
        ///   Instantiates an all-zeros variable and returns it.
        /// </summary>
        /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
        /// <param name="dtype">Data type of returned Keras variable.</param>
        /// <param name="name">String, name of returned Keras variable.</param>
        /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
        public Tensor zeros(int[] shape, DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            // The following is not necessary since C# is strongly typed:
            // shape = tuple(map(int, shape))
            // tf_dtype = _convert_string_dtype(dtype)

            // However, we might have to perform other conversions of our own:
            Type type = TFTensor.TypeFromTensorType(In(dtype.Value));
            Array zeros = Array.CreateInstance(type, shape);

            return this.variable(array: zeros, name: name);
        }

        /// <summary>
        ///   Element-wise equality between two tensors.
        /// </summary>
        /// 
        /// <param name="x">Tensor or variable.</param>
        /// <param name="y">Tensor or variable.</param>
        /// 
        /// <returns>A bool tensor.</returns>
        /// 
        public Tensor equal(Tensor x, Tensor y)
        {
            return Out(tf.Equal(In(x), In(y)));
        }

        /// <summary>
        ///   Returns the index of the maximum value along an axis.
        /// </summary>
        /// 
        /// <param name="x">Tensor or variable.</param>
        /// <param name="axis">The axis along which to perform the reduction.</param>
        /// 
        /// <returns>A tensor.</returns>
        /// 
        public Tensor argmax(Tensor x, int axis = -1)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L1332
            //axis = _normalize_axis(axis, ndim(x));
            return Out(tf.ArgMax(In(x), tf.Const(axis)));
        }

        public Tensor round(Tensor x)
        {
            return Out(tf.Round(In(x)));
        }

        public DataType floatx()
        {
            return DataType.Float;
        }






        #region conversion

        public TFShape In(int?[] shape)
        {
            return new TFShape(shape.Select(x => x.HasValue ? (long)x.Value : -1).ToArray());
        }

        public Tensor Out(TFOutput output)
        {
            return new TensorFlowTensor(this) { output = output };
        }

        public Tensor Out(TFTensor output)
        {
            return Out(tf.Const(output));
        }

        public TensorFlowTensor In(Tensor output)
        {
            return (TensorFlowTensor)output;
        }

        public TensorFlowTensor In(TFOutput output)
        {
            return new TensorFlowTensor(this) { output = output };
        }

        public static TFDataType In(DataType dataType)
        {
            return (TFDataType)dataType;
        }

        public static TFDataType? In(DataType? dataType)
        {
            if (dataType == null)
                return null;
            return (TFDataType)dataType.Value;
        }

        public static DataType? Out(TFDataType? dataType)
        {
            if (dataType == null)
                return null;
            return (DataType)dataType.Value;
        }

        public static DataType Out(TFDataType dataType)
        {
            return (DataType)dataType;
        }

        #endregion





        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                    if (tf != null)
                        tf.Dispose();
                    if (_SESSION != null)
                        _SESSION.Dispose();
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
                tf = null;
                _SESSION = null;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        // ~TensorFlowBackend() {
        //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        //   Dispose(false);
        // }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion
    }
}
