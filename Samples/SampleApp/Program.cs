using Accord.Math;
using CNTK;
using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Backends;
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

            // Let's use CNTK for this example
            KerasSharp.Backends.Current.Switch("KerasSharp.Backends.CNTKBackend");

            // Load the Pima Indians Data Set
            var pima = new Accord.DataSets.PimaIndiansDiabetes();
            double[,] x = pima.Instances.ToMatrix();
            double[] y = pima.ClassLabels.ToDouble();

            // Create the model
            var model = new Sequential();
            model.Add(new Dense(12, input_dim: 8, activation: new ReLU()));
            model.Add(new Dense(8, activation: new ReLU()));
            model.Add(new Dense(1, activation: new Sigmoid()));

            // Compile the model
            model.Compile(loss: new BinaryCrossEntropy(), optimizer: new Adam(), metrics: new Accuracy());

            // Fit the model
            model.fit(x, y, epochs: 150, batch_size: 10);

            // Evaluate the model
            double[,] pred = model.predict(x, batch_size: 10)[0].To<double[,]>();

            Console.ReadLine();
        }
    }
}
