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
    using static KerasSharp.Python;
    using System.Diagnostics;
    using Accord;

    [DataContract]
    [DebuggerDisplay("{ToString()}")]
    public abstract class Tensor : IConvertible
    {
        public IBackend K;
        public int?[] _keras_shape;
        public bool _uses_learning_phase;
        public int?[] int_shape;
        public (Layer layer, int node_index, int tensor_index)? _keras_history
        {
            get;
            set;
        }

        public string name;

        public DataType? dtype
        {
            get { return K.dtype(this); }
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

        public object eval()
        {
            return K.eval(this);
        }

        public object eval<T>()
        {
            return eval().To<T>();
        }

        public TypeCode GetTypeCode()
        {
            throw new NotImplementedException();
        }

        public bool ToBoolean(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public char ToChar(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public sbyte ToSByte(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public byte ToByte(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public short ToInt16(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public ushort ToUInt16(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public int ToInt32(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public uint ToUInt32(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public long ToInt64(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public ulong ToUInt64(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public float ToSingle(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public double ToDouble(IFormatProvider provider)
        {
            return (double)eval<double>();
        }

        public decimal ToDecimal(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public DateTime ToDateTime(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public string ToString(IFormatProvider provider)
        {
            throw new NotImplementedException();
        }

        public object ToType(Type conversionType, IFormatProvider provider)
        {
            throw new NotImplementedException();
        }


        // TODO: Generate these operators using T4 templates

        public static Tensor operator *(double a, Tensor b)
        {
            return b.K.mul(a, b);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return b.K.mul(a, b);
        }

        public static Tensor operator /(double a, Tensor b)
        {
            return b.K.div(a, b);
        }

        public static Tensor operator /(Tensor a, Tensor b)
        {
            return b.K.div(a, b);
        }

        public static Tensor operator +(double a, Tensor b)
        {
            return b.K.add(a, b);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return b.K.add(a, b);
        }

        public static Tensor operator +(Tensor a, double b)
        {
            return a.K.add(a, b);
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
