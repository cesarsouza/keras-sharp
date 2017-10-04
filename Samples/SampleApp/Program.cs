using Accord.Math;
using KerasSharp;
using KerasSharp.Backends;
using KerasSharp.Initializers;
using KerasSharp.Models;
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
            var model = new Sequential();
            var dense = new Dense(units: 5, input_dim: 5,
                kernel_initializer: new Constant(Matrix.Identity(5)),
                bias_initializer: new Constant(0));
            model.Add(dense);

            float[,] input = Vector.Range(25).Reshape(5, 5).ToSingle();
            float[,] output = MatrixEx.To<float[,]>(model.predict(input)[0]);

        }
    }
}
