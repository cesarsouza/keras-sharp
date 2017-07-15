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
    using TensorFlow;
    using KerasSharp.Models;

    public interface IBackend
    {
        // TODO: Rename all methods to PascalCase

        Tensor sqrt(object p);

        Tensor square(Tensor w);

        Tensor sum(Tensor v, int axis, bool keepdims);

        Tensor clip(Tensor norms, int v, int maxValue);

        Tensor epsilon();

        TFDataType floatx();

        Tensor greater_equal(Tensor w, double v);

        Tensor cast(object v1, object v2);

        Tensor dropout(object p, double retain_prob, object noise_shape, object seed);

        Tensor relu(Tensor x);

        Tensor softmax(Tensor x);


        Tensor max(Tensor x, int v, object p);

        int ndim(Tensor x);

        Tensor max(Tensor x, int axis, bool keepdims);

        Tensor div(Tensor e, Tensor s);

        Tensor elu(Tensor x);

        Tensor hard_sigmoid(Tensor x);

        Tensor mul(double scale, Tensor tensor);

        Tensor elu(Tensor x, double alpha);

        Tensor sigmoid(Tensor x);

        Tensor subtract(Tensor x, Tensor tensor);

        Tensor softplus(Tensor x);

        Tensor softsign(Tensor x);

        Tensor tanh(Tensor x);

        Tensor exp(object v);

        Tensor div(Tensor desired, object v);


        Tensor add(Tensor desired, Tensor v);

        Tensor mul(Tensor w, Tensor tensor);

        Tensor clip(Tensor norms, double min_value, double max_value);

        Tensor add(double v, Tensor tensor);

        Tensor constant(int v, TFShape shape, TFDataType dtype);

        Tensor random_uniform(TFShape shape, double v, double limit, TFDataType dtype, int? seed);

        Tensor truncated_normal(TFShape shape, double v, double stddev, TFDataType dtype, int? seed);

        Tensor l2_normalize(Tensor expected, int axis);

        Tensor minus(Tensor tensor);

        Tensor mean(Tensor tensor, int axis);

        Tensor const_(int v);

        Tensor abs(Tensor input);

        Tensor categorical_crossentropy(Tensor expected, Tensor actual);

        Tensor sum(Tensor tensor, int axis);

        Tensor max(Tensor tensor, int axis);

        Tensor maximum(double v, Tensor tensor);

        Tensor elu(object x);

        Tensor binary_crossentropy(Tensor expected, Tensor actual);

        double subtract(double v, Tensor expected);

        Tensor constant(int v, int?[] shape, TFDataType dtype);

        Tensor add(object v1, double v2);

        Tensor variable(double v, string name);

        Tensor in_train_phase(Func<Tensor> dropped_inputs, Tensor inputs, bool? training);

        TFDataType? dtype(Tensor input_tensor);

        Tensor constant(int v, int[] shape, TFDataType dtype);

        Tensor placeholder(int[] shape, TFDataType? dtype, bool sparse, string name);

        string get_uid(string prefix);

        List<Tensor> gradients(ILoss loss, object param);

        int?[] int_shape(Tensor input_tensor);

        Tensor variable(Tensor Tensor, TFDataType? dtype, string name);

        object sum(object[] v);

        IDisposable name_scope(string name);

        object sum(Tensor tensor);

        Tensor clip_norm(Tensor g, double clipnorm, Tensor norm);

        Tensor identity(Tensor x);

        List<Array> batch_get_value(List<Tensor> weights);

        void batch_set_value(List<Tuple<Tensor, Array>> weight_value_tuples);

        Tensor placeholder(int?[] shape, TFDataType? dtype, bool sparse, string name);

        int?[] int_shape(TFTensor input_tensor);

        Tensor div(double v1, object v2);

        List<Array> batch_get_value(List<List<Tensor>> weights);

        void batch_set_value(List<(Tensor, Array)> tuples);

        Tensor update_add(Tensor iterations, int v);

        Tensor placeholder(int ndim, string name);

        Tensor mul(double rate, Tensor tensor1, Tensor tensor2);

        Tensor add(Tensor tensor);

        Tensor mul(double rate, double v, Tensor tensor);

        object get_variable_shape(Tensor p);

        Tensor get_variable_shape(object s);

        object sum(double v, Tensor tensor);

        bool is_sparse(Tensor tensor);

        Tensor placeholder(int ndim, string name, bool sparse, TFDataType? dtype);

        Tensor add(object total_loss, object v);

        Tensor const_(double v);

        object learning_phase();

        Function function(object inputs, List<Tensor> list, Func<List<object>> updates, string name);

        Tensor mul(Tensor momentum, object v);

        Function function<TSource>(List<Tensor> inputs, List<object> list, List<TSource> updates, string name);

        Function function(List<Tensor> inputs, List<object> list, Func<List<object>> updates, string name);

        Tensor update(object m, object v);

        double mul(Tensor batch_out, int length);

        Tensor mul(List<Tensor> batch_outs, int length);

        Tensor div(Tensor tensor, int samples);

        Tensor truncated_normal(int[] shape, double v, double stddev, TFDataType dtype, int? seed);

        Tensor random_uniform(int[] shape, double v, double limit, TFDataType dtype, int? seed);

        Tensor truncated_normal(int?[] shape, double v, double stddev, TFDataType dtype, int? seed);

        Tensor random_uniform(int?[] shape, double v, double limit, TFDataType dtype, int? seed);
    }
}
