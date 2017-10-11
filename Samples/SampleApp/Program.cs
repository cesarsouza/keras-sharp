using Accord.Math;
using CNTK;
using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Backends;
using KerasSharp.Metrics;
using KerasSharp.Models;
using System;
using System.Collections;
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
            var iris = new Accord.DataSets.Iris();
            double[,] x = iris.Instances.ToMatrix();
            double[,] y = Matrix.OneHot(iris.ClassLabels);

            KerasSharp.Backends.Current.Switch("KerasSharp.Backends.CNTKBackend");

            // For a single-input model with 2 classes (binary classification):

            var model = new Sequential();
            model.Add(new Dense(10, input_dim: 4, activation: new ReLU()));
            model.Add(new Dense(3));
            model.Compile(optimizer: "sgd",
                          loss: "mse",
                          metrics: new [] { new Accuracy() });

            // Train the model, iterating on the data in batches of 32 samples
            model.fit(x, y, epochs: 10, batch_size: 10);

            // Use the model to predict the class labels
            double[,] pred = MatrixEx.To<double[,]>(model.predict(x, batch_size: 10)[0]);
        }
    }
}
