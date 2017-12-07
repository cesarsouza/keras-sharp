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
    using static KerasSharp.Python;
    using Accord.Math;
    using C = CNTK.CNTKLib;
    using CNTK;
    using Accord.Math.Comparers;
    using Accord;
    using CNTKDataType = CNTK.DataType;
    using DataType = KerasSharp.DataType;
    using System.Runtime.CompilerServices;
    using System.Diagnostics;

    public class BackendCall
    {
        public string Member { get; set; }
        public string Source { get; set; }
        public int Line { get; set; }
        public List<string> Parameters { get; set; }
        public List<object> Values { get; set; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append($"K.{Member}(");
            sb.Append(String.Join("    ", Enumerable.Zip(Parameters, Values, (a, b) => (a, b)).Select((p, v) => $"{p}: {v}")));
            sb.Append(")");
            return sb.ToString();
        }
    }

    public partial class CNTKBackend : BackendBase, IBackend
    {
        public List<BackendCall> events;
        Stack<string> NAME_SCOPE_STACK;
        Dictionary<string, int> _UID_PREFIXES;

        // cntk doesn't support gradient as symbolic op, to hook up with keras model,
        // we will create gradient as a constant placeholder, here use this global
        // map to keep the mapping from grad placeholder to parameter
        internal Dictionary<string, Parameter> grad_parameter_dict;


        public CNTKBackend()
        {
            events = new List<BackendCall>();
            this.NAME_SCOPE_STACK = new Stack<string>();
            this._UID_PREFIXES = new Dictionary<string, int>();
            this.grad_parameter_dict = new Dictionary<string, Parameter>();

            //C.SetTraceLevel(CNTK.TraceLevel.Info);
            //C.SetMathLibTraceLevel(1);
        }

        [Conditional("Disabled")]
        private void log<T>(T item,
            [CallerMemberName] string memberName = "",
            [CallerFilePath] string sourceFilePath = "",
            [CallerLineNumber] int sourceLineNumber = 0)
           where T : class
        {
            //var properties = typeof(T).GetProperties();

            //if (typeof(T).FullName.StartsWith("CNTK"))
            //{
            //    if (item is CNTK.Function)
            //    {
            //        OnFunction(item as CNTK.Function);
            //    }

            //    if (item is Variable)
            //    {
            //        OnVariable(item as Variable);
            //    }
            //}

            //events.Add(new BackendCall
            //{
            //    Member = memberName,
            //    Line = sourceLineNumber,
            //    Parameters = properties.Select(x => x.Name).ToList(),
            //    Values = properties.Select(x => x.GetValue(item)).ToList(),
            //});

            //return item;
        }

        private static void OnFunction(CNTK.Function v)
        {
            foreach (var vv in v.Outputs)
            {
                //if (vv.Uid.Contains("Reshape6"))
                //    throw new Exception();
                //if (vv.Uid.Contains("142"))
                //    throw new Exception();
                //if (vv.Uid.Contains("143"))
                //    throw new Exception();
            }
        }

        private static void OnVariable(Variable v)
        {
            if (v.Name.Contains("Reshape"))
                throw new Exception();

            if (v.Uid.Contains("Reshape"))
                throw new Exception();

        }

        public Tensor sqrt(Tensor x)
        {
            log(new { x });
            return Out(C.Sqrt(In(x)));
        }

        public Tensor pow(Tensor x, Tensor p, string name = null)
        {
            log(new { x, p });
            if (name == null)
                return Out(C.Pow(In(x), In(p)));
            return Out(C.Pow(In(x), In(p), name: name));
        }

        public Tensor square(Tensor w)
        {
            log(new { w });
            return Out(C.Square(In(w)));
        }

        public Tensor equal(Tensor x, Tensor y)
        {
            log(new { x, y });
            return Out(C.Equal(In(x), In(y)));
        }

        public Tensor sum(List<Tensor> x, int[] axis = null, bool keepdims = false, string name = null)
        {
            log(new { x, axis, keepdims, name });
            throw new NotImplementedException();
        }

        public Tensor sum(Tensor x, int[] axis = null, bool keepdims = false, string name = null)
        {
            return _reduce(x, axis, keepdims, C.ReduceSum);
        }

        private Tensor _reduce(Tensor x, int[] axis, bool keepdims, Func<Variable, AxisVector, CNTK.Function> func)
        {
            var _x = In(x);

            Axis[] _axis;

            if (axis == null)
                _axis = new[] { Axis.AllAxes() };

            _axis = axis.Select(a => new Axis(a)).ToArray(); // Axes in reduce operations are 1-based (?)

            CNTK.Function f = _x;
            if (axis.Length > 0)
                f = func(_x, new AxisVector(_axis));

            f = _remove_dims(f, axis, keepdims);
            return Out(f);
        }

        public Tensor round(Tensor x)
        {
            return Out(C.Round(In(x)));
        }

        public Tensor argmax(Tensor x, int axis = -1)
        {
            log(new { x, axis });

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L745
            var _axis = new Axis(axis);
            var _x = In(x);

            CNTK.Function output = C.Argmax(_x.function, _axis);
            output = _reshape_dummy_dim(output, axis);
            return Out(output);
        }

        private CNTK.Function _reshape_dummy_dim(CNTK.Function x, params int[] axis)
        {
            log(new { x, axis });

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L680

            List<int> shape = In(x.Output.Shape).ToList();


            var _axis = axis.Select(i => i < 0 ? (i + shape.Count) : i).ToArray();

            if (shape.Count(s => s == NDShape.InferredDimension) > 1)
            {
                var result = x;
                foreach (int index in _axis.Sorted().Reverse())
                {
                    result = C.Reshape(result, replacementShape: NDShape.CreateNDShape(new int[] { }),
                        beginAxis: new Axis(index), endAxis: new Axis(index + 1));
                }
                return result;
            }
            else
            {
                foreach (int index in _axis.Sorted().Reverse())
                    shape.RemoveAt(index);

                if (shape.Count == 1 && shape[0] < 0)
                    shape[0] = NDShape.InferredDimension;
                return C.Reshape(x, NDShape.CreateNDShape(shape));
            }
        }

        public Tensor sum(Tensor x, int axis, bool keepdims = false, string name = null)
        {
            log(new { x, axis, keepdims, name });
            return sum(x, new[] { axis }, keepdims, name);
        }

        public Tensor clip(Tensor norms, int v, int maxValue)
        {
            throw new NotImplementedException();
        }

        public Tensor print_tensor(Tensor x, string message)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L1778
            Console.WriteLine("Not implemented yet.");
            return x;
        }

        public Tensor zeros(int[] shape, DataType? dtype = null, string name = null)
        {
            log(new { shape, dtype, name });

            if (dtype == null)
                dtype = floatx();

            return Out(new Parameter(InShape(shape), In(dtype.Value), 0.0));
        }

        public Tensor zeros(int?[] shape, DataType? dtype, string name = null)
        {
            log(new { shape, dtype, name });
            return zeros(shape.Select(x => x.Value).ToArray(), dtype, name);
        }

        public Tensor greater_equal(Tensor w, double v)
        {
            throw new NotImplementedException();
        }

        public void clear_session()
        {
            this.NAME_SCOPE_STACK.Clear();
            this._UID_PREFIXES.Clear();
        }

        public Tensor cast(Tensor x, DataType dataType)
        {
            // cntk calculates everything in float, so don't need case from bool / int
            return x;
        }

        public Tensor dropout(object p, double retain_prob, object noise_shape, object seed)
        {
            throw new NotImplementedException();
        }

        public Tensor relu(Tensor x)
        {
            log(new { x });
            return Out(C.ReLU(In(x).function));
        }

        public Tensor softmax(Tensor x)
        {
            log(new { x });
            return Out(C.Softmax(In(x).function));
        }

        public Tensor max(Tensor x, int v, object p)
        {
            throw new NotImplementedException();
        }

        public int? ndim(Tensor x)
        {
            return In(x).function.Output.Shape.Rank;
        }

        public Tensor max(Tensor x, int axis, bool keepdims)
        {
            throw new NotImplementedException();
        }

        public Tensor elu(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor hard_sigmoid(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor mul(Tensor a, Tensor b, string name = null)
        {
            if (name == null)
                return Out(C.ElementTimes(In(a), In(b)));
            return Out(C.ElementTimes(In(a), In(b), name: name));
        }

        public Tensor mul<T>(T a, Tensor b, string name = null)
        {
            if (name == null)
                return Out(C.ElementTimes(InGeneric(a), In(b)));
            return Out(C.ElementTimes(InGeneric(a), In(b), name: name));
        }

        public Tensor mul<T>(Tensor a, T b, string name = null)
        {
            if (name == null)
                return Out(C.ElementTimes(In(a), InGeneric(b)));
            return Out(C.ElementTimes(In(a), InGeneric(b), name: name));
        }

        public Tensor mul(List<Tensor> batch_outs, int length)
        {
            throw new NotImplementedException();
        }

        public Tensor div(Tensor a, Tensor b)
        {
            return Out(C.ElementDivide(In(a), In(b)));
        }

        public Tensor div<T>(T a, Tensor b)
        {
            return Out(C.ElementDivide(InGeneric(a), In(b)));
        }

        public Tensor div<T>(Tensor a, T b)
        {
            return Out(C.ElementDivide(In(a), InGeneric(b)));
        }

        public Tensor add(Tensor a, Tensor b)
        {
            return Out(new Variable(In(a).function) + new Variable(In(b).function));
        }

        public Tensor bias_add(Tensor output, Tensor bias, DataFormatType? data_format = null, string name = null)
        {
            if (data_format != null)
                throw new NotImplementedException();

            using (this.name_scope("bias_add"))
            {
                CNTKTensor _x = In(output);
                CNTKTensor _b = In(bias);
                var _shape = In(_x.CNTK_Shape).Select(x => x < 0 ? 1 : x).ToArray();

                var shape = NDShape.CreateNDShape(_shape);

                var b = C.Reshape(_b, shape);

                return Out(new Variable(_x.function) + b);
            }
        }

        public Tensor add<T>(Tensor a, T b)
        {
            return Out(In(a) + InGeneric(b));
        }

        public Tensor add<T>(T a, Tensor b)
        {
            return Out(InGeneric(a) + In(b));
        }

        public Tensor subtract(Tensor a, Tensor b, string name = null)
        {
            if (name == null)
                return Out(C.Minus(In(a), In(b)));
            return Out(C.Minus(In(a), In(b), name: name));
        }

        public Tensor subtract<T>(Tensor a, T b, string name = null)
        {
            return Out(C.Minus(In(a), InGeneric(b), name: name));
        }

        public Tensor subtract<T>(T a, Tensor b, string name = null)
        {
            if (name == null)
                return Out(C.Minus(InGeneric(a), In(b)));
            return Out(C.Minus(InGeneric(a), In(b), name: name));
        }

        public Tensor dot(Tensor a, Tensor b, string name = null)
        {
            if (name == null)
                return Out(C.Times(In(a), In(b)));
            return Out(C.Times(In(a), In(b), name: name));
        }

        public Tensor elu(Tensor x, double alpha)
        {
            throw new NotImplementedException();
        }

        public Tensor sigmoid(Tensor x)
        {
            return Out(C.Sigmoid(In(x)));
        }

        public Tensor softplus(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor softsign(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor tanh(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor exp(Tensor x)
        {
            return Out(C.Exp(In(x)));
        }

        public Tensor transpose(Tensor tensor)
        {
            return Out(C.Transpose(In(tensor)));
        }

        /// <summary>
        ///   Turn a nD tensor into a 2D tensor with same 0th dimension. In other words, it flattens each data samples of a batch.
        /// </summary>
        /// 
        public Tensor batch_flatten(Tensor x)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L1460
            // cntk's batch axis is not in shape,
            // so just flatten all the dim in x.shape
            int dim = Matrix.Product(x.shape.Select(s => s.Value).ToArray());
            x = Out(C.Reshape(In(x), NDShape.CreateNDShape(new[] { -1 })));
            x._keras_shape = new int?[] { null, dim };
            return x;
        }

        public object eval(Tensor tensor)
        {
            log(new { tensor });

            CNTKTensor _tensor = In(tensor);
            Variable variable = _tensor.function;
            var inputs = new Dictionary<Variable, Value>();
            var outputs = new Dictionary<Variable, Value>()
            {
                { variable, null }
            };
            _tensor.function.Evaluate(inputs, outputs, DeviceDescriptor.CPUDevice);
            Value value = outputs[variable];
            var shape = value.Shape;

            IEnumerable<IEnumerable<double>> r = 
                value.GetDenseData<double>(_tensor.function);

            if (value.DataType == CNTKDataType.Double)
                return Out<double>(variable, value, shape);
            if (value.DataType == CNTKDataType.Float)
                return Out<float>(variable, value, shape);
            throw new InvalidOperationException();
        }

        public Tensor clip(Tensor norms, double minval, double maxval)
        {
            throw new NotImplementedException();
        }

        public Tensor random_uniform(int[] shape, double minval = 0, double maxval = 1, DataType? dtype = null, int? seed = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            if (seed == null)
            {
                // ensure that randomness is conditioned by the Accord RNG
                seed = Accord.Math.Random.Generator.Random.Next(10_000);
            }

            return Out(C.UniformRandom(shape: InShape(shape), dataType: In(dtype.Value),
                low: minval, high: maxval, seed: (uint)seed.Value, name: In(name)));
        }

        public Tensor l2_normalize(Tensor expected, int axis)
        {
            throw new NotImplementedException();
        }

        public Tensor minus(Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor mean(Tensor tensor, int axis = -1, bool keepdims = false, string name = null)
        {
            return mean(tensor, new[] { axis }, keepdims, name);
        }

        public Tensor mean(Tensor x, int[] axis, bool keepdims = false, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L701
            return _reduce(x, axis, keepdims, C.ReduceMean);
        }

        private CNTK.Function _remove_dims(CNTK.Function x, int[] axis, bool keepdims = false)
        {
            log(new { x, axis, keepdims });

            using (this.name_scope("_remove_dims"))
            {
                if (keepdims == false)
                    return _reshape_dummy_dim(x, axis);

                return x;
            }
        }

        public Tensor abs(Tensor input)
        {
            throw new NotImplementedException();
        }

        public Tensor categorical_crossentropy(Tensor target, Tensor output, bool from_logits = false)
        {
            log(new { target, output, from_logits });

            using (this.name_scope("categorical_crossentropy"))
            {
                // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L1480

                var _output = In(output);
                var _target = In(target);

                if (from_logits)
                {
                    var result = C.CrossEntropyWithSoftmax(_output, _target);
                    // cntk's result shape is (batch, 1), while keras expect (batch, )
                    CNTK.Function r = C.Reshape(result, NDShape.CreateNDShape(new int[] { }));
                    return Out(r);
                }
                else
                {
                    // scale preds so that the class probas of each sample sum to 1
                    var o = C.ElementDivide(_output.function, C.ReduceSum(_output, Axis.EndStaticAxis()));
                    var eps = Constant.Scalar(epsilon(), DeviceDescriptor.CPUDevice);
                    var omeps = Constant.Scalar(1.0 - epsilon(), DeviceDescriptor.CPUDevice);
                    // avoid numerical instability with _EPSILON clipping
                    o = C.Clip(o, eps, omeps);
                    CNTK.Function r = C.Negate(C.ReduceSum(C.ElementTimes(_target, C.Log(_output)), Axis.EndStaticAxis()));
                    return Out(r);
                }
            }
        }

        public Tensor max(Tensor tensor, int axis)
        {
            throw new NotImplementedException();
        }

        public Tensor maximum(double v, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public Tensor elu(object x)
        {
            throw new NotImplementedException();
        }

        public Tensor binary_crossentropy(Tensor output, Tensor target, bool from_logits = false)
        {
            log(new { output, target, from_logits });

            var _output = new Variable(In(output).function);
            var _target = new Variable(In(target).function);

            if (from_logits)
                _output = C.Sigmoid(_output);

            // scale preds so that the class probas of each sample sum to 1
            var eps = InConstant(epsilon());
            var omeps = InConstant(1.0);
            // avoid numerical instability with _EPSILON clipping
            _output = C.Clip(_output, eps, omeps);
            var a = new Variable(C.Negate(C.ElementTimes(_target, C.Log(_output))));
            var b = new Variable(C.Negate(
                C.Minus(C.ElementTimes(InConstant(1.0), _target),
                C.Minus(C.Log(InConstant(1.0)), _output))));
            _output = a + b;
            return Out(_output);
        }

        private string _prepare_name(string name, string _default = "")
        {
            string prefix = String.Join("_", NAME_SCOPE_STACK);
            if (String.IsNullOrEmpty(name))
            {
                if (String.IsNullOrEmpty(prefix))
                    return _default;
                return prefix + '/' + _default;
            }
            else
            {
                if (String.IsNullOrEmpty(prefix))
                    return name;
                return prefix + '/' + name;
            }
        }

        public Tensor variable(Array array, DataType? dtype = null, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L133

            NDArrayView v = In(array, dtype);

            var p = new Parameter(v, name: _prepare_name(name, "variable"));

            Tensor t = Out(p);
            return t;
        }

        public Tensor variable<T>(T value, DataType? dtype, string name = null)
            where T : struct
        {
            return variable((Array)new T[] { value }, dtype, name: name);
        }

        public Tensor variable(Tensor tensor, DataType? dtype = null, string name = null)
        {
            log(new { tensor, dtype, name });

            if (dtype == null)
                dtype = floatx();

            object v = tensor.eval();

            if (v is Array)
            {
                return variable((Array)v, dtype, name);
            }
            else
            {
                Array r = Array.CreateInstance(v.GetType(), 1);
                r.SetValue(v, 0);
                return variable(r, dtype, name);
            }
        }

        public Tensor in_train_phase(Func<Tensor> x, Func<Tensor> alt, bool? training)
        {
            throw new NotImplementedException();
        }

        public KerasSharp.DataType? dtype(Tensor input_tensor)
        {
            return Out(In(input_tensor).function.Output.DataType);
        }

        public Tensor constant<T>(T value, int[] shape = null, KerasSharp.DataType? dtype = null, string name = null)
        {
            log(new { value, shape, dtype, name });

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py#L305

            Constant _const = InGeneric(value, shape, dtype, name);

            return Out(_const);
        }

        public Constant InGeneric<T>(T value, int[] shape = null, KerasSharp.DataType? dtype = null, string name = null)
        {
            if (dtype == null)
                dtype = floatx();

            CNTKDataType _dtype = In(dtype.Value);

            int[] _shape;
            if (shape == null)
            {
                if (value is Array)
                {
                    _shape = (value as Array).GetLength();
                }
                else
                {
                    _shape = new int[] { };
                }
            }
            else
            {
                _shape = shape;
            }

            Constant c = _constant(value, _shape, _dtype, name);

            if (!Out(c.Shape).IsEqual(_shape))
                throw new Exception();

            if (dtype == null || c.DataType == _dtype)
                return c;

            throw new Exception();
        }

        public Constant _constant<T>(T value, int[] shape, CNTKDataType dtype, string name)
        {
            if (value is Array)
            {
                NDArrayView x = In(value as Array, Out(dtype));
                if (name != null)
                    return new Constant(x, name);
                else return new Constant(x);
            }

            if (shape == null || shape.Length == 0)
            {
                if (dtype == CNTKDataType.Double)
                    return Constant.Scalar<double>(value.To<double>(), device: DeviceDescriptor.CPUDevice);
                if (dtype == CNTKDataType.Float)
                    return Constant.Scalar<float>(value.To<float>(), device: DeviceDescriptor.CPUDevice);
            }

            if (name == null)
            {
                return new Constant(shape: InShape(shape), dataType: dtype,
                    initValue: (dynamic)value, device: DeviceDescriptor.CPUDevice);
            }

            return new Constant(shape: InShape(shape), dataType: dtype,
                initValue: (dynamic)value, device: DeviceDescriptor.CPUDevice, name: name);
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
            if (!_UID_PREFIXES.ContainsKey(prefix))
                _UID_PREFIXES[prefix] = 0;
            _UID_PREFIXES[prefix] += 1;
            return _UID_PREFIXES[prefix];
        }

        public List<Tensor> gradients(Tensor loss, List<Tensor> variables)
        {
            log(new { loss, variables });

            // cntk does not support gradients as symbolic op,
            // to hook up with keras model
            // we will return a constant as place holder, the cntk learner will apply
            // the gradient during training.

            var grads = new List<Constant>();
            foreach (Tensor t in variables)
            {
                Parameter v = In(t).parameter;
                Constant g = new Constant(shape: v.Shape, dataType: CNTKDataType.Double, initValue: 0.0, device: DeviceDescriptor.CPUDevice, name: "keras_grad_placeholder");
                grads.Add(g);
                grad_parameter_dict[g.Uid] = v;
            };

            return grads.Select(g => Out(g)).ToList();
        }

        public int?[] int_shape(Tensor tensor)
        {
            log(new { tensor });

            NDShape shape = In(tensor).function.Output.Shape;
            int?[] r = new int?[shape.Rank];
            for (int i = 0; i < r.Length; i++)
            {
                if (shape[i] >= 0)
                    r[i] = shape[i];
            }

            return r;
        }

        public object sum(object[] v)
        {
            throw new NotImplementedException();
        }

        public NameScope name_scope(string name)
        {
            return new CNTKNameScope(NAME_SCOPE_STACK, name);
        }

        public Tensor clip_norm(Tensor g, double clipnorm, Tensor norm)
        {
            throw new NotImplementedException();
        }

        public Tensor identity(Tensor x, string name = null)
        {
            return Out(C.Alias(In(x), _prepare_name(name)));
        }

        public List<Array> batch_get_value(List<Tensor> weights)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<Tuple<Tensor, Array>> weight_value_tuples)
        {
            throw new NotImplementedException();
        }

        public Tensor placeholder(int?[] shape = null, int? ndim = null, DataType? dtype = null, bool sparse = false, string name = null)
        {
            log(new { shape, ndim, dtype, sparse, name });

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/cntk_backend.py
            if (shape == null)
            {
                if (ndim != null)
                    shape = new int?[ndim.Value];
            }

            var cntk_shape = shape.Select(s => s == null ? NDShape.FreeDimension : s.Value);

            //if (dynamic_axis_num > len(cntk_shape)
            //{
            //    raise ValueError('CNTK backend: creating placeholder with '
            //            '%d dimension is not supported, at least '
            //            '%d dimensions are needed.'
            //            % (len(cntk_shape, dynamic_axis_num)))
            //}

            if (dtype == null)
                dtype = floatx();

            if (name is null)
                name = String.Empty;

            // cntk_shape = cntk_shape[dynamic_axis_num:]

            NDShape _shape = NDShape.CreateNDShape(cntk_shape);
            CNTKDataType _dtype = In(dtype.Value);
            Variable v = Variable.InputVariable(shape: _shape, dataType: _dtype, isSparse: sparse, name: name);

            var x = Out(v);

            x._keras_shape = shape;
            x._uses_learning_phase = false;
            return x;
        }

        public List<Array> batch_get_value(List<List<Tensor>> weights)
        {
            throw new NotImplementedException();
        }

        public void batch_set_value(List<(Tensor, Array)> tuples)
        {
            throw new NotImplementedException();
        }

        public Tensor update_add<T>(Tensor x, T increment, string name) where T : struct
        {
            CNTKTensor _x = In(x);
            CNTK.Function result = _x + InGeneric(increment);
            return Out(C.Assign(_x, result, name: name));
        }

        public int?[] get_variable_shape(Tensor x)
        {
            return Out(In(x).CNTK_Shape);
        }

        public Tensor sum(double v, Tensor tensor)
        {
            throw new NotImplementedException();
        }

        public bool is_sparse(Tensor tensor)
        {
            log(new { tensor });
            return In(tensor).function.Output.IsSparse;
        }

        public object learning_phase()
        {
            throw new NotImplementedException();
        }

        public Models.Function function(List<Tensor> inputs, List<Tensor> list, List<List<Tensor>> updates, string name)
        {
            Variable[] _inputs = inputs.Select(x => (Variable)In(x).function).ToArray();
            Variable[] _outputs = list.Select(x => (Variable)In(x).function).ToArray();
            //CNTK.Function[] _list = list.Select(x => In(x).function).ToArray();
            return new CNTKFunction(this, _inputs, _outputs, updates, name);
        }

        public Models.Function function(object inputs, List<Tensor> list, List<Tensor> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Models.Function function<TSource>(List<Tensor> inputs, List<object> list, List<TSource> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Models.Function function(List<Tensor> inputs, List<object> list, Func<List<object>> updates, string name)
        {
            throw new NotImplementedException();
        }

        public Tensor update(Tensor x, Tensor new_x, string name = null)
        {
            return Out(C.Assign(In(x), In(new_x), name: name));
        }

        public Tensor truncated_normal(int[] shape, double v, double stddev, DataType? dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor truncated_normal(int?[] shape, double v, double stddev, DataType? dtype, int? seed)
        {
            throw new NotImplementedException();
        }

        public Tensor not_equal<T>(Tensor weights, T v) where T : struct
        {
            return Out(C.NotEqual(In(weights), InConstant((dynamic)v)));
        }

        public Tensor reshape(Tensor x, int[] shape)
        {
            log(new { x, shape });
            return Out(C.Reshape(In(x).function, InShape(shape)));
        }


        public Tensor conv1d(Tensor inputs, Tensor kernel, int strides, PaddingType padding, DataFormatType? data_format = null, int dilation_rate = 1, string name = null)
        {
            throw new NotImplementedException();
        }

        public Tensor conv2d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format = null, int[] dilation_rate = null, string name = null)
        {
            throw new NotImplementedException();
        }

        public Tensor conv3d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format = null, int[] dilation_rate = null, string name = null)
        {
            throw new NotImplementedException();
        }



        public DataType floatx()
        {
            return DataType.Double;
        }



        #region conversion

        private static Axis[] InAxis(int[] axis)
        {
            if (axis == null)
                return new[] { Axis.AllAxes() };

            return axis.Select(a => new Axis(a)).ToArray();
        }

        public static string In(string name)
        {
            if (name == null)
                return String.Empty;
            return name;
        }

        public static Constant InConstant(double value)
        {
            return Constant.Scalar<double>(value, DeviceDescriptor.CPUDevice);
        }

        public NDArrayView In(Array array, DataType? dtype = null)
        {
            if (dtype == null)
                dtype = this.floatx();

            Type t = array.GetInnerMostType();
            if (t != dtype.Value.ToType())
                return In(array.Convert(dtype.Value.ToType()), dtype);

            int[] shape = array.GetLength();
            NDShape cntk_shape = NDShape.CreateNDShape(shape);

            Array dataBuffer = array.Flatten();

            // cntk will init type based on the value type
            NDArrayView v;
            if (dataBuffer.GetInnerMostType() == typeof(double))
            {
                v = new NDArrayView(cntk_shape, (double[])dataBuffer, DeviceDescriptor.CPUDevice);
            }
            else if (dataBuffer.GetInnerMostType() == typeof(float))
            {
                v = new NDArrayView(cntk_shape, (float[])dataBuffer, DeviceDescriptor.CPUDevice);
            }
            else
            {
                throw new InvalidOperationException("Execution should never reach here.");
            }

            return v;
        }

        public Tensor Out(Value value)
        {
            return Out(new Constant(value.Data));
        }

        public static object Out<T>(Variable variable, Value value, NDShape shape)
        {
            IList<IList<T>> r = value.GetDenseData<T>(variable);

            if (shape.Rank == 0)
                return r[0][0];

            var rr = r.Apply(c => c.ToArray().Reshape(In(shape)));

            if (rr.Length == 1)
                return rr[0];
            return rr;
        }

        public NDShape InShape(int?[] shape)
        {
            return NDShape.CreateNDShape(shape.Select(x => x.HasValue ? (int)x.Value : -1));
        }

        public NDShape InShape(int[] shape)
        {
            return NDShape.CreateNDShape(shape);
        }

        public int?[] Out(NDShape shape)
        {
            int?[] s = new int?[shape.Rank];
            for (int i = 0; i < s.Length; i++)
            {
                if (shape[i] >= 0)
                    s[i] = shape[i];
            }

            return s;
        }

        public Tensor Out(CNTK.Function function, int[] keras_shape)
        {
            return Out(function, keras_shape.Select(x => (int?)x).ToArray());
        }

        public Tensor Out(CNTK.Function function, int?[] keras_shape = null)
        {
            var t = new CNTKTensor(this)
            {
                function = function
            };
            if (keras_shape == null)
                keras_shape = t.shape;
            t._keras_shape = keras_shape;
            t._uses_learning_phase = false;
            return t;
        }

        public Tensor Out(CNTK.Parameter parameter, int?[] keras_shape = null)
        {
            var t = new CNTKTensor(this)
            {
                function = parameter.ToFunction(),
                parameter = parameter
            };
            if (keras_shape == null)
                keras_shape = t.shape;
            t._keras_shape = keras_shape;
            t._uses_learning_phase = false;
            return t;
        }


        public CNTKTensor In(Tensor output)
        {
            return (CNTKTensor)output;
        }

        public static CNTKDataType? In(DataType? dataType)
        {
            if (dataType == null)
                return null;
            return (CNTKDataType)dataType.Value;
        }

        public static DataType? Out(CNTKDataType? dataType)
        {
            if (dataType == null)
                return null;
            return (DataType)dataType.Value;
        }

        public static CNTKDataType In(DataType dataType)
        {
            switch (dataType)
            {
                case DataType.Float:
                    return CNTKDataType.Float;
                case DataType.Double:
                    return CNTKDataType.Double;
                case DataType.UInt8:
                    return CNTKDataType.UChar;
            }

            throw new ArgumentOutOfRangeException(nameof(dataType));
        }

        public static DataType Out(CNTKDataType dataType)
        {
            switch (dataType)
            {
                case CNTKDataType.Float:
                    return DataType.Float;
                case CNTKDataType.Double:
                    return DataType.Double;
                case CNTKDataType.UChar:
                    return DataType.UInt8;
            }

            throw new ArgumentOutOfRangeException(nameof(dataType));
        }

        private static int[] In(NDShape shape)
        {
            int[] r = new int[shape.Rank];
            for (int i = 0; i < r.Length; i++)
                r[i] = shape[i];
            return r;
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
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
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
