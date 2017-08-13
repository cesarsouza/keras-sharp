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

    public interface IBackend : IDisposable
    {
        // TODO: Rename all methods to PascalCase

        Tensor sqrt(object p);

        Tensor square(Tensor w);

        Tensor sum(Tensor v, int axis, bool keepdims);

        Tensor clip(Tensor norms, int v, int maxValue);

        Tensor epsilon();

        TFDataType floatx();

        Tensor greater_equal(Tensor w, double v);
        void clear_session();
        Tensor cast(object v1, object v2);

        Tensor dropout(object p, double retain_prob, object noise_shape, object seed);

        Tensor relu(Tensor x);

        Tensor softmax(Tensor x);


        Tensor max(Tensor x, int v, object p);

        int? ndim(Tensor x);

        Tensor max(Tensor x, int axis, bool keepdims);

        Tensor div(Tensor e, Tensor s);

        Tensor elu(Tensor x);

        Tensor hard_sigmoid(Tensor x);



        Tensor mul<T>(T a, Tensor b);

        Tensor mul<T>(Tensor a, T b);

        Tensor mul(List<Tensor> batch_outs, int length);



        Tensor add(Tensor a, Tensor b);

        Tensor add<T>(Tensor a, T b);

        Tensor add<T>(T a, Tensor b);



        Tensor elu(Tensor x, double alpha);

        Tensor sigmoid(Tensor x);

        Tensor subtract(Tensor x, Tensor tensor);

        Tensor softplus(Tensor x);

        Tensor softsign(Tensor x);

        Tensor tanh(Tensor x);

        Tensor exp(object v);

        Tensor div(Tensor desired, object v);

        object eval(Tensor tensor);

        

        Tensor mul(Tensor w, Tensor tensor);

        Tensor clip(Tensor norms, double min_value, double max_value);

        

        Tensor random_uniform(int?[] shape, double minvalue = 0.0, double maxvalue = 1.0, TFDataType dtype = Utils.DEFAULT_DTYPE, int? seed = null, string name = null);

        Tensor truncated_normal(TFShape shape, double v, double stddev, TFDataType dtype, int? seed);

        Tensor l2_normalize(Tensor expected, int axis);

        Tensor minus(Tensor tensor);

        Tensor mean(Tensor tensor, int axis);

        Tensor abs(Tensor input);

        Tensor categorical_crossentropy(Tensor expected, Tensor actual);

        Tensor sum(Tensor tensor, int axis);

        Tensor max(Tensor tensor, int axis);

        Tensor maximum(double v, Tensor tensor);

        Tensor elu(object x);

        Tensor binary_crossentropy(Tensor expected, Tensor actual);

        double subtract(double v, Tensor expected);



        Tensor variable(Array array, string name = null);

        Tensor variable<T>(T value, string name = null) where T : struct;

        Tensor variable(Tensor tensor, TFDataType dtype = Utils.DEFAULT_DTYPE, string name = null);

        Tensor in_train_phase(Func<Tensor> dropped_inputs, Tensor inputs, bool? training);

        TFDataType? dtype(Tensor input_tensor);

        Tensor constant<T>(T value, int?[] shape = null, TFDataType? dtype = null, string name = null);

        int get_uid(string prefix);

        List<Tensor> gradients(Tensor loss, object param);

        int?[] int_shape(Tensor input_tensor);


        object sum(object[] v);

        IDisposable name_scope(string name);

        object sum(Tensor tensor);

        Tensor clip_norm(Tensor g, double clipnorm, Tensor norm);

        Tensor identity(Tensor x);

        List<Array> batch_get_value(List<Tensor> weights);

        void batch_set_value(List<Tuple<Tensor, Array>> weight_value_tuples);

        Tensor placeholder(int?[] shape, TFDataType? dtype = Utils.DEFAULT_DTYPE, bool sparse = false, string name = null);

        int?[] int_shape(TFTensor input_tensor);

        Tensor div(double v1, object v2);

        List<Array> batch_get_value(List<List<Tensor>> weights);

        void batch_set_value(List<(Tensor, Array)> tuples);

        Tensor update_add(Tensor iterations, int v);

        Tensor placeholder(int ndim, string name);

        Tensor add(Tensor tensor);

        object get_variable_shape(Tensor p);

        Tensor get_variable_shape(object s);

        object sum(double v, Tensor tensor);

        bool is_sparse(Tensor tensor);

        Tensor placeholder(int ndim, string name, bool sparse, TFDataType? dtype);

        Tensor add(object total_loss, object v);

        object learning_phase();

        Function function(object inputs, List<Tensor> list, Func<List<List<Tensor>>> updates, string name);

        Function function(object inputs, List<Tensor> list, List<List<Tensor>> updates, string name);


        Function function<TSource>(List<Tensor> inputs, List<object> list, List<TSource> updates, string name);

        Function function(List<Tensor> inputs, List<object> list, Func<List<object>> updates, string name);

        Tensor update(object m, object v);

        Tensor div(Tensor tensor, int samples);

        Tensor truncated_normal(int[] shape, double v, double stddev, TFDataType dtype, int? seed);

        Tensor truncated_normal(int?[] shape, double v, double stddev, TFDataType dtype, int? seed);

    }
}
