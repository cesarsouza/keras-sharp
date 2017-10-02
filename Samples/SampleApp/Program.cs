using CNTK;
using KerasSharp.Backends;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SampleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var K = new CNTKBackend())
            {
                var input = K.placeholder(shape: new int?[] { 2, 4, 5 });
                double[,] val = new double[,] { { 1, 2 }, { 3, 4 } };
                var kvar = K.variable(array: (Array)val);
                int? a = K.ndim(input); // 3
                int? b = K.ndim(kvar); // 2
            }
        }
    }
}
