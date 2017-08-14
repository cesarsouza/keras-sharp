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

namespace KerasSharp.Engine.Topology
{
    using Accord.Math;
    using KerasSharp.Backends;
    using KerasSharp.Layers;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;

    [DataContract]
    public class Tensor
    {
        public IBackend K;
        public TFTensor tensor;
        public TFOutput output;
        public int?[] _keras_shape;
        public bool _uses_learning_phase;
        public int?[] int_shape;
        public (Layer layer, int node_index, int tensor_index)? _keras_history;
        public string name;

        public TFDataType dtype
        {
            get
            {
                if (tensor != null)
                    return tensor.TensorType;
                return output.OutputType;
            }
        }

        public Tensor(IBackend backend)
        {
            this.K = backend;
        }

        public static IEnumerable<Tensor> Zero { get; internal set; }
        public static Tensor One { get; internal set; }

        internal static Tensor Zeros(int?[] shape, object dtype)
        {
            throw new NotImplementedException();
        }

        public int?[] shape
        {
            get { return K.int_shape(this); }
        }

        internal long[] TF_Shape
        {
            get
            {
                var tf = (K as TensorFlowBackend).tf;
                return tf.GetShape(output);
            }
        }


        public object eval()
        {
            return K.eval(this);
        }


        // TODO: Generate these operators automatically

        public static Tensor operator *(double a, Tensor b)
        {
            return b.K.mul(a, b);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return b.K.mul(a, b);
        }

        public static Tensor operator +(double a, Tensor b)
        {
            return b.K.add(a, b);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return b.K.add(a, b);
        }

        public static Tensor operator -(double a, Tensor b)
        {
            return b.K.subtract(a, b);
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            return b.K.subtract(a, b);
        }
    }

}
