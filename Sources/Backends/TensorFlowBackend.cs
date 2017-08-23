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
    using Accord.Math;
    using static KerasSharp.Python;

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
            return tensor(tf.Reshape(x, constant(shape)));
        }



        public Tensor abs(Tensor input)
        {
            throw new NotImplementedException();
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

        public Tensor binary_crossentropy(Tensor expected, Tensor actual)
        {
            throw new NotImplementedException();
        }

        public Tensor cast(Tensor x, TFDataType dataType)
        {
            return tensor(tf.Cast(x, dataType));
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
            // Note: tf.nn.softmax_cross_entropy_with_logits
            // expects logits, Keras expects probabilities.
            if (!from_logits)
            {
                // scale preds so that the class probas of each sample sum to 1
                int?[] shape = output.shape;
                var last = tf.Const(new TFTensor(shape.Length - 1));
                TFOutput o = tf.Div(output, tf.ReduceSum(output, axis: last, keep_dims: true));
                // manual computation of crossentropy
                TFOutput _epsilon = constant(epsilon(), dtype: output.dtype);
                o = tf.ClipByValue(o, _epsilon, tf.Sub(constant(1f), _epsilon));
                o = tf.Neg(tf.ReduceSum(tf.Mul(target, tf.Log(output)), axis: last));
                return tensor(o);
            }

            return tensor(tf.SoftmaxCrossEntropyWithLogits(target, output).loss);
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

        public Tensor constant<T>(T value, int?[] shape = null, TFDataType? dtype = null, string name = null)
        {
            TFTensor t = new TFTensor((dynamic)value);

            TFOutput o = dtype == null ?
                tf.Const(t, operName: name) :
                tf.Const(t, dtype: dtype.Value, operName: name);

            return tensor(o);
        }





        public Tensor dropout(object p, double retain_prob, object noise_shape, object seed)
        {
            throw new NotImplementedException();
        }

        public TFDataType? dtype(Tensor input_tensor)
        {
            return input_tensor.dtype;
        }

        public Tensor elu(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor elu(Tensor x, double alpha)
        {
            throw new NotImplementedException();
        }

        public Tensor elu(object x)
        {
            throw new NotImplementedException();
        }

        public Tensor exp(object v)
        {
            throw new NotImplementedException();
        }

        public TFDataType floatx()
        {
            return TFDataType.Float;
        }

        public Function function(object inputs, List<Tensor> list, Func<List<object>> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Function function<TSource>(List<Tensor> inputs, List<object> list, List<TSource> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Function function(List<Tensor> inputs, List<object> list, Func<List<object>> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Function function(object inputs, List<Tensor> list, Func<List<Tensor>> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Function function(object inputs, List<Tensor> list, List<Tensor> updates, string name)
        {
            throw new NotImplementedException();
        }




        public object get_variable_shape(Tensor p)
        {
            throw new NotImplementedException();
        }

        public Tensor get_variable_shape(object s)
        {
            throw new NotImplementedException();
        }

        public List<Tensor> gradients(ILoss loss, object param)
        {
            throw new NotImplementedException();
        }

        public List<Tensor> gradients(Tensor loss, object param)
        {
            throw new NotImplementedException();
        }

        public Tensor greater_equal(Tensor w, double v)
        {
            throw new NotImplementedException();
        }

        public Tensor not_equal(Tensor x, Tensor y)
        {
            return tensor(tf.NotEqual(x, y));
        }

        public Tensor not_equal(Tensor x, double y)
        {
            return tensor(tf.NotEqual(x, tf.Const(y, x.dtype)));
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

            try
            {
                long[] shape = tf.GetTensorShape(x.output);
                return shape.Apply(i => i == -1 ? null : (int?)i);
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

        public Tensor in_train_phase(Func<Tensor> dropped_inputs, Tensor inputs, bool? training)
        {
            throw new NotImplementedException();
        }

        public bool is_sparse(Tensor tensor)
        {
            return false;
        }

        public Tensor l2_normalize(Tensor expected, int axis)
        {
            throw new NotImplementedException();
        }

        public object learning_phase()
        {
            throw new NotImplementedException();
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
            return tensor(tf.ReduceMean(x, _normalize_axis(axis, ndim(x)), keepdims, operName: name));
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
            return tensor(tf.ReduceMean(x, axis: tf.Const(axis), keep_dims: keepdims, operName: name));
        }


        public Tensor minus(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor dot(Tensor a, Tensor b)
        {
            return tensor(tf.MatMul(a.output, b.output));
        }


        public Tensor mul<T>(T a, Tensor b)
        {
            return mul(constant(a, dtype: b.dtype), b);
        }

        public Tensor mul(Tensor a, Tensor b)
        {
            return tensor(tf.Mul(a.output, b.output));
        }

        public Tensor mul<T>(Tensor a, T b)
        {
            return mul(a, constant(b, dtype: a.dtype));
        }

        public Tensor mul(List<Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }





        public Tensor div<T>(T a, Tensor b)
        {
            return div(constant(a, dtype: b.dtype), b);
        }

        public Tensor div(Tensor a, Tensor b)
        {
            return tensor(tf.Mul(a.output, b.output));
        }

        public Tensor div<T>(Tensor a, T b)
        {
            return div(a, constant(b, dtype: a.dtype));
        }

        public Tensor div(List<Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }



        public Tensor add(Tensor a, Tensor b)
        {
            return tensor(tf.Add(a.output, b.output));
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
            return tensor(tf.Sub(a.output, b.output));
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

            return tf.GetTensorNumDims(x.output);
        }

        public Tensor placeholder(int?[] shape = null, int? ndim = null, TFDataType? dtype = Utils.DEFAULT_DTYPE, bool sparse = false, string name = null)
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

            var tfshape = this.shape(shape);

            Tensor x = tensor(tf.Placeholder(dtype.Value, tfshape, operName: name));
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
        public Tensor random_uniform(int?[] shape, double minval = 0.0, double maxval = 1.0, TFDataType dtype = Utils.DEFAULT_DTYPE, int? seed = null, string name = null)
        {
            if (seed == null)
                seed = Accord.Math.Random.Generator.Random.Next(1_000_000);

            var tf_shape = tf.Const(shape.Apply(x => (long)x));
            TFOutput u = tf.RandomUniform(tf_shape, dtype: dtype, seed: seed, operName: name);


            return tensor(tf.Add(tf.Mul(u, tf.Const(new TFTensor(maxval - minval), dtype: dtype)),
                                        tf.Const(new TFTensor(minval), dtype: dtype)));
        }

        public Tensor relu(Tensor x)
        {
            return tensor(tf.Relu(x));
        }

        public Tensor sigmoid(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor softmax(Tensor x)
        {
            return tensor(tf.Softmax(x.output));
        }

        public Tensor softplus(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor softsign(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor sqrt(object p)
        {
            throw new NotImplementedException();
        }

        public Tensor square(Tensor w)
        {
            throw new NotImplementedException();
        }

        public Tensor sum(Tensor x, int[] axis, bool keepdims = false, string name = null)
        {
            return tensor(tf.ReduceSum(x, tf.Const(axis), keepdims, name));
        }

        public Tensor sum(Tensor x, int axis, bool keepdims = false, string name = null)
        {
            return tensor(tf.ReduceSum(x, tf.Const(axis), keepdims, name));
        }

        public object sum(object[] v)
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

        public Tensor truncated_normal(TFShape shape, double v, double stddev, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor truncated_normal(int[] shape, double v, double stddev, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor truncated_normal(int?[] shape, double v, double stddev, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor update(object m, object v)
        {
            throw new NotImplementedException();
        }

        public Tensor update_add(Tensor iterations, int v)
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

            Tensor t = new Tensor(this);
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

            Tensor t = new Tensor(this);
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
        public Tensor variable(Tensor tensor, TFDataType dtype = Utils.DEFAULT_DTYPE, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308

            Tensor t = new Tensor(this);
            if (tensor.tensor == null)
                t.output = tensor.output;
            else
                t.output = tf.Const(tensor.tensor);
            t._keras_shape = tensor.shape;
            t._uses_learning_phase = false;
            return t;
        }

        public object eval(Tensor tensor)
        {
            TFTensor[] result = _SESSION.Run(new TFOutput[] { }, new TFTensor[] { }, new[] { tensor.output });

            if (result.Length == 1)
                return result[0].GetValue();

            return result.Apply(x => x.GetValue());
        }


        public TFShape shape(int?[] shape)
        {
            return new TFShape(shape.Select(x => x.HasValue ? (long)x.Value : -1).ToArray());
        }

        public Tensor tensor(TFOutput output)
        {
            return new Tensor(this) { output = output };
        }


        /// <summary>
        ///   Instantiates an all-zeros variable and returns it.
        /// </summary>
        /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
        /// <param name="dtype">Data type of returned Keras variable.</param>
        /// <param name="name">String, name of returned Keras variable.</param>
        /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
        public Tensor zeros(int[] shape, TFDataType dtype = Utils.DEFAULT_DTYPE, string name = null)
        {
            // The following is not necessary since C# is strongly typed:
            // if dtype is None:
            //     dtype = floatx()
            // shape = tuple(map(int, shape))
            // tf_dtype = _convert_string_dtype(dtype)

            // However, we might have to perform other conversions of our own:
            Type type = Utils.GetSystemType(dtype);
            Array zeros = Array.CreateInstance(type, shape);

            return this.variable(array: zeros, name: name);
        }






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
            return tensor(tf.Equal(x, y));
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
            return tensor(tf.ArgMax(x, tf.Const(axis)));
        }
        #endregion
    }
}
