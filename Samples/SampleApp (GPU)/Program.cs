using Accord;
using Accord.Math;
using Accord.Statistics.Analysis;
using CNTK;
using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Backends;
using KerasSharp.Initializers;
using KerasSharp.Losses;
using KerasSharp.Metrics;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using static KerasSharp.Backends.Current;

namespace SampleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            # Example from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

            from keras.models import Sequential
            from keras.layers import Dense
            import numpy

            # fix random seed for reproducibility
            numpy.random.seed(7)

            # load pima indians dataset
            dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
            # split into input (X) and output (Y) variables
            X = dataset[:,0:8]
            Y = dataset[:,8]

            # create model
            model = Sequential()
            model.add(Dense(12, input_dim=8, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Fit the model
            model.fit(X, Y, epochs=150, batch_size=10)

            # evaluate the model
            scores = model.evaluate(X, Y)
            print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            */

            //KerasSharp.Backends.Current.Switch("KerasSharp.Backends.TensorFlowBackend");
            KerasSharp.Backends.Current.Switch("KerasSharp.Backends.CNTKBackend");

            // Load the Pima Indians Data Set
            var pima = new Accord.DataSets.PimaIndiansDiabetes();
            float[,] x = pima.Instances.ToMatrix().ToSingle();
            float[] y = pima.ClassLabels.ToSingle();

            // Create the model
            var model = new Sequential();
            model.Add(new Dense(12, input_dim: 8, activation: new ReLU()));
            model.Add(new Dense(8, activation: new ReLU()));
            model.Add(new Dense(1, activation: new Sigmoid()));

            // Compile the model (for the moment, only the mean square 
            // error loss is supported, but this should be solved soon)
            model.Compile(loss: new MeanSquareError(), 
                optimizer: new Adam(), 
                metrics: new[] { new Accuracy() });

            // Fit the model for 150 epochs
            model.fit(x, y, epochs: 150, batch_size: 10);

            // Use the model to make predictions
            float[] pred = model.predict(x)[0].To<float[]>();

            // Evaluate the model
            double[] scores = model.evaluate(x, y);
            Console.WriteLine($"{model.metrics_names[1]}: {scores[1] * 100}");

            Console.ReadLine();
        }
    }
}
