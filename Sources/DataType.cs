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

using Accord.Math;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace KerasSharp
{
    public enum DataType : uint
    {
        Float = 1,
        Double = 2,
        Int32 = 3,
        UInt8 = 4,
        Int16 = 5,
        Int8 = 6,
        String = 7,
        Complex64 = 8,
        Complex = 8,
        Int64 = 9,
        Bool = 10,
        QInt8 = 11,
        QUInt8 = 12,
        QInt32 = 13,
        BFloat16 = 14,
        QInt16 = 15,
        QUInt16 = 16,
        UInt16 = 17,
        Complex128 = 18,
        Half = 19,
        Resource = 20
    }

    public static class DataTypeExtensions
    {
        public static Type ToType(this DataType? type)
        {
            return type?.ToType();
        }

        public static Type ToType(this DataType type)
        {
            switch (type)
            {
                case DataType.Float:
                    return typeof(float);
                case DataType.Double:
                    return typeof(double);
            }

            throw new ArgumentOutOfRangeException(nameof(type));
        }
    }
}
