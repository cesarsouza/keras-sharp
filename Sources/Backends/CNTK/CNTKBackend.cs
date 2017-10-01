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
    using C = CNTK.CNTKLib;
    using CNTK;

    public class CNTKBackend : BackendBase, IBackend<CNTKTensor>
    {

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py

        public CNTKBackend()
        {
        }

        private CNTKTensor tensor(CNTK.Function function)
        {
            return new CNTKTensor(function);
        }

        public CNTKTensor abs(CNTKTensor input)
        {
            return tensor(C.Abs(input.variable));
        }

        public CNTKTensor add(CNTKTensor a, CNTKTensor b)
        {
            return tensor(a.variable + b.variable);
        }

        public CNTKTensor add<T>(CNTKTensor a, T b)
        {
            return add(a, constant(b));
        }

        public CNTKTensor add<T>(T a, CNTKTensor b)
        {
            return add(constant(a), b);
        }

        public CNTKTensor argmax(CNTKTensor x, int axis = -1)
        {
            return tensor(C.Argmax(x.variable, new Axis(axis)));
        }

        public List<Array> batch_get_value(List<CNTKTensor> weights)
        {
            throw new NotImplementedException();
        }

        public List<Array> batch_get_value(List<List<CNTKTensor>> weights)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<Tuple<CNTKTensor, Array>> weight_value_tuples)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<(CNTKTensor, Array)> tuples)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor binary_crossentropy(CNTKTensor output, CNTKTensor target, bool from_logits = false)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor cast(CNTKTensor x, TFDataType dataType)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor categorical_crossentropy(CNTKTensor target, CNTKTensor output, bool from_logits = false)
        {
            throw new NotImplementedException();
        }

        public void clear_session()
        {
            throw new NotImplementedException();
        }

        public CNTKTensor clip(CNTKTensor norms, int v, int maxValue)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor clip(CNTKTensor norms, double min_value, double max_value)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor clip_norm(CNTKTensor g, double clipnorm, CNTKTensor norm)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor constant<T>(T value, int?[] shape = null, TFDataType? dtype = null, string name = null)
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public CNTKTensor div(CNTKTensor a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor div<T>(T a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor div<T>(CNTKTensor a, T b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor dot(CNTKTensor a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor dropout(object p, double retain_prob, object noise_shape, object seed)
        {
            throw new NotImplementedException();
        }

        public TFDataType? dtype(CNTKTensor input_tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor elu(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor elu(CNTKTensor x, double alpha)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor elu(object x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor equal(CNTKTensor x, CNTKTensor y)
        {
            throw new NotImplementedException();
        }

        public object eval(CNTKTensor tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor exp(object v)
        {
            throw new NotImplementedException();
        }

        public TFDataType floatx()
        {
            throw new NotImplementedException();
        }

        public Models.Function function(object inputs, List<CNTKTensor> list, Func<List<CNTKTensor>> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Models.Function function(object inputs, List<CNTKTensor> list, List<CNTKTensor> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Models.Function function<TSource>(List<CNTKTensor> inputs, List<object> list, List<TSource> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Models.Function function(List<CNTKTensor> inputs, List<object> list, Func<List<object>> updates, string name)
        {
            throw new NotImplementedException();
        }

        public int get_uid(string prefix)
        {
            throw new NotImplementedException();
        }

        public int?[] get_variable_shape(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public List<CNTKTensor> gradients(CNTKTensor loss, List<CNTKTensor> param)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor greater_equal(CNTKTensor w, double v)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor hard_sigmoid(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor identity(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public int?[] int_shape(CNTKTensor tensor)
        {
            throw new NotImplementedException();
        }

        public int?[] int_shape(TFTensor input_tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor in_train_phase(Func<CNTKTensor> x, Func<CNTKTensor> alt, bool? training)
        {
            throw new NotImplementedException();
        }

        public bool is_sparse(CNTKTensor tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor l2_normalize(CNTKTensor expected, int axis)
        {
            throw new NotImplementedException();
        }

        public object learning_phase()
        {
            throw new NotImplementedException();
        }

        public CNTKTensor max(CNTKTensor x, int v, object p)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor max(CNTKTensor x, int axis, bool keepdims)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor max(CNTKTensor tensor, int axis)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor maximum(double v, CNTKTensor tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor mean(CNTKTensor tensor, int axis = -1, bool keepdims = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor mean(CNTKTensor tensor, int[] axis, bool keepdims = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor minus(CNTKTensor tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor mul(CNTKTensor a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor mul<T>(T a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor mul<T>(CNTKTensor a, T b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor mul(List<CNTKTensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }

        public IDisposable name_scope(string name)
        {
            throw new NotImplementedException();
        }

        public int? ndim(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor not_equal(CNTKTensor weights, double v)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor placeholder(int?[] shape = null, int? ndim = null, TFDataType? dtype = TFDataType.Float, bool sparse = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor random_uniform(int?[] shape, double minvalue = 0, double maxvalue = 1, TFDataType dtype = TFDataType.Float, int? seed = null, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor relu(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor round(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor sigmoid(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor softmax(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor softplus(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor softsign(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor sqrt(object p)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor square(CNTKTensor w)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor subtract(CNTKTensor a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor subtract<T>(CNTKTensor a, T b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor subtract<T>(T a, CNTKTensor b)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor sum(CNTKTensor x, int[] axis = null, bool keepdims = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor sum(CNTKTensor x, int axis, bool keepdims = false, string name = null)
        {
            throw new NotImplementedException();
        }

        public object sum(object[] v)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor sum(double v, CNTKTensor tensor)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor tanh(CNTKTensor x)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor truncated_normal(TFShape shape, double v, double stddev, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor truncated_normal(int[] shape, double v, double stddev, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor truncated_normal(int?[] shape, double v, double stddev, TFDataType dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor update(object m, object v)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor update_add(CNTKTensor iterations, int v)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor variable(Array array, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor variable<T>(T value, string name = null) where T : struct
        {
            throw new NotImplementedException();
        }

        public CNTKTensor variable(CNTKTensor tensor, TFDataType dtype = TFDataType.Float, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor zeros(int[] shape, TFDataType dtype = TFDataType.Float, string name = null)
        {
            throw new NotImplementedException();
        }

        public CNTKTensor zeros(int?[] shape, TFDataType dtype = TFDataType.Float, string name = null)
        {
            throw new NotImplementedException();
        }
    }
}
