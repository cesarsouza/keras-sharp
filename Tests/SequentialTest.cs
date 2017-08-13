using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Losses;
using KerasSharp.Metrics;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Tests
{
    /// <summary>
    ///   Test for examples in https://keras.io/models/sequential/
    /// </summary>
    /// 
    [TestFixture]
    public class SequentialTest
    {
        [Test]
        public void sequential_example_1()
        {
            #region doc_sequential_example_1
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 500 }));
            model.Add(new Dense(10, activation: new Softmax()));
            model.Compile(optimizer: new RootMeanSquareProp(),
                  loss: new CategoricalCrossEntropy(),
                  metrics: new Accuracy());

            #endregion

            Assert.AreEqual(0, model.trainable_weights);
        }

        [TearDown]
        public void TearDown()
        {
            KerasSharp.Backends.Current.K.clear_session();
        }
    }
}
