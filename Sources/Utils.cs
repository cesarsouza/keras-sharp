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

namespace KerasSharp
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;

    public static class Utils
    {
        public const TFDataType DEFAULT_DTYPE = TFDataType.Float;

        public static long GetLastDimension(TFShape shape)
        {
            return shape[shape.NumDimensions - 1];
        }

        public static Type GetSystemType(TFDataType dtype)
        {
            switch (dtype)
            {
                case TFDataType.Float:
                    return typeof(float);
                case TFDataType.Double:
                    return typeof(double);
                case TFDataType.Int32:
                    return typeof(Int32);
                case TFDataType.UInt8:
                    return typeof(Byte);
                case TFDataType.Int16:
                    break;
                case TFDataType.Int8:
                    break;
                case TFDataType.String:
                    break;
                case TFDataType.Complex64:
                    break;
                //case TFDataType.Complex:
                //    break;
                case TFDataType.Int64:
                    break;
                case TFDataType.Bool:
                    break;
                case TFDataType.QInt8:
                    break;
                case TFDataType.QUInt8:
                    break;
                case TFDataType.QInt32:
                    break;
                case TFDataType.BFloat16:
                    break;
                case TFDataType.QInt16:
                    break;
                case TFDataType.QUInt16:
                    break;
                case TFDataType.UInt16:
                    break;
                case TFDataType.Complex128:
                    break;
                case TFDataType.Half:
                    break;
                case TFDataType.Resource:
                    break;
            }

            throw new NotSupportedException();
        }

        internal static object ToNetType(object dtype)
        {
            throw new NotImplementedException();
        }
    }
}
