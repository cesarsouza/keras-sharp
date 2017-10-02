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
    using System.Threading;
    using System.Reflection;

    public static class Current
    {
        private static ThreadLocal<IBackend> backend;

        private static string[] assemblyNames =
        {
            "KerasSharp.Backends.TensorFlow",
            "KerasSharp.Backends.CNTK.CPUOnly",
        };

        public static string Name = "KerasSharp.Backends.TensorFlowBackend";

        public static IBackend K
        {
            get { return backend.Value; }
            set { backend.Value = value; }
        }

        static Current()
        {
            backend = new ThreadLocal<IBackend>(() => load(Name));
        }

        public static void Switch(string backendName)
        {
            Name = backendName;
            backend.Value = load(Name);
        }



        private static IBackend load(string typeName)
        {
            Type type = find(typeName);

            IBackend obj = (IBackend)Activator.CreateInstance(type);

            return obj;
        }

        private static Type find(string typeName)
        {
            foreach (string assemblyName in assemblyNames)
            {
                Assembly assembly = Assembly.Load(assemblyName);

                var types = assembly.GetExportedTypes();

                foreach (var type in types)
                {
                    string currentTypeName = type.FullName;
                    if (currentTypeName == typeName)
                        return type;
                }
            }

            throw new ArgumentException("typeName");
        }

    }
}
