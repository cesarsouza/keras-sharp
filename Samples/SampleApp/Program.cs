using CNTK;
using KerasSharp;
using KerasSharp.Backends;
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
            KerasSharp.Backends.Current.Switch<CNTKBackend>();

            // For a single-input model with 2 classes (binary classification):

            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_dim: 100));
            model.Add(new Dense(1, activation: "sigmoid"));
            model.Compile(optimizer: "rmsprop",
                          loss: "binary_crossentropy",
                          metrics: new[] { "accuracy" });

            // Generate dummy data
            double[,] data = Accord.Math.Matrix.Random(1000, 100);
            int[] labels = Accord.Math.Vector.Random(1000, min: 0, max: 10);

            // Train the model, iterating on the data in batches of 32 samples
            model.fit(data, labels, epochs: 10, batch_size: 32);
            // For a single-input model with 10 classes (categorical classification):
        }
    }
}
