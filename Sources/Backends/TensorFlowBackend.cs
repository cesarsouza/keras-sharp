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

    // TODO:

    internal class TensorFlowBackend : IBackend
    {
        public Tensor abs(Tensor input)
        {
            throw new NotImplementedException();
        }

        public Tensor add(Tensor desired, Tensor v)
        {
            throw new NotImplementedException();
        }

        public Tensor add(double v, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor add(object v1, double v2)
        {
            throw new NotImplementedException();
        }

        public Tensor add(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor add(object total_loss, object v)
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

        public Tensor cast(object v1, object v2)
        {
            throw new NotImplementedException();
        }

        public Tensor categorical_crossentropy(Tensor expected, Tensor actual)
        {
            throw new NotImplementedException();
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

        public Tensor constant(int v, TFShape shape, TFDataType dtype)
        {
            throw new NotImplementedException();
        }

        public Tensor constant(int v, int?[] shape, TFDataType dtype)
        {
            throw new NotImplementedException();
        }

        public Tensor constant(int v, int[] shape, TFDataType dtype)
        {
            throw new NotImplementedException();
        }

        public Tensor const_(int v)
        {
            throw new NotImplementedException();
        }

        public Tensor const_(double v)
        {
            throw new NotImplementedException();
        }

        public Tensor div(Tensor e, Tensor s)
        {
            throw new NotImplementedException();
        }

        public Tensor div(Tensor desired, object v)
        {
            throw new NotImplementedException();
        }

        public Tensor div(double v1, object v2)
        {
            throw new NotImplementedException();
        }

        public Tensor div(Tensor tensor, int samples)
        {
            throw new NotImplementedException();
        }

        public Tensor dropout(object p, double retain_prob, object noise_shape, object seed)
        {
            throw new NotImplementedException();
        }

        public TFDataType? dtype(Tensor input_tensor)
        {
            throw new NotImplementedException();
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

        public Tensor epsilon()
        {
            throw new NotImplementedException();
        }

        public Tensor exp(object v)
        {
            throw new NotImplementedException();
        }

        public TFDataType floatx()
        {
            throw new NotImplementedException();
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

        public string get_uid(string prefix)
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

        public Tensor greater_equal(Tensor w, double v)
        {
            throw new NotImplementedException();
        }

        public Tensor hard_sigmoid(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor identity(Tensor x)
        {
            throw new NotImplementedException();
        }

        public int?[] int_shape(Tensor input_tensor)
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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

        public Tensor mean(Tensor tensor, int axis)
        {
            throw new NotImplementedException();
        }

        public Tensor minus(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(double scale, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(Tensor w, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(double rate, Tensor tensor1, Tensor tensor2)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(double rate, double v, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(Tensor momentum, object v)
        {
            throw new NotImplementedException();
        }

        public double mul(Tensor batch_out, int length)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(List<Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }

        public IDisposable name_scope(string name)
        {
            throw new NotImplementedException();
        }

        public int ndim(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor placeholder(int[] shape, TFDataType? dtype, bool sparse, string name)
        {
            throw new NotImplementedException();
        }

        public Tensor placeholder(int?[] shape, TFDataType? dtype, bool sparse, string name)
        {
            throw new NotImplementedException();
        }

        public Tensor placeholder(int ndim, string name)
        {
            throw new NotImplementedException();
        }

        public Tensor placeholder(int ndim, string name, bool sparse, TFDataType? dtype)
        {
            throw new NotImplementedException();
        }

        public Tensor random_uniform(TFShape shape, double v, double limit, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor random_uniform(int[] shape, double v, double limit, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor random_uniform(int?[] shape, double v, double limit, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor relu(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor sigmoid(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor softmax(Tensor x)
        {
            throw new NotImplementedException();
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

        public Tensor subtract(Tensor x, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public double subtract(double v, Tensor expected)
        {
            throw new NotImplementedException();
        }

        public Tensor sum(Tensor v, int axis, bool keepdims)
        {
            throw new NotImplementedException();
        }

        public Tensor sum(Tensor tensor, int axis)
        {
            throw new NotImplementedException();
        }

        public object sum(object[] v)
        {
            throw new NotImplementedException();
        }

        public object sum(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public object sum(double v, Tensor tensor)
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

        public Tensor variable(double v, string name)
        {
            throw new NotImplementedException();
        }

        public Tensor variable(Tensor Tensor, TFDataType? dtype, string name)
        {
            throw new NotImplementedException();
        }
    }
}
