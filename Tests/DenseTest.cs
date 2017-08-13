using KerasSharp;
using KerasSharp.Losses;
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
    [TestFixture]
    public class DenseTest
    {
        [Test]
        public void dense_example_1()
        {
            // First example from https://keras.io/layers/core/#dense

            #region doc_dense_example_1
            // as first layer in a sequential model:
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 16 }));
            // now the model will take as input arrays of shape (*, 16)
            // and output arrays of shape (*, 32)

            // after the first layer, you don't need to specify
            // the size of the input anymore:
            model.Add(new Dense(32));
            #endregion

            Assert.AreEqual(2, model.layers.Count);
            Assert.AreEqual("dense_1", model.layers[0].name);
            Assert.AreEqual(new int?[] { null, 16 }, model.layers[0].input_shape[0]);
            Assert.AreEqual("dense_2", model.layers[1].name);
            Assert.AreEqual(new int?[] { null, 32 }, model.layers[1].input_shape[0]);
        }

        [Test]
        public void compile_test()
        {
            #region doc_dense_example_1
            // as first layer in a sequential model:
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 16 }));
            model.Add(new Dense(32));

            model.Compile(new StochasticGradientDescent(), new BinaryCrossEntropy());

            #endregion

            Assert.AreEqual(0, model.trainable_weights);
        }

        [SetUp]
        protected void SetUp()
        {
        }

        [TearDown]
        public void TearDown()
        {
            KerasSharp.Backends.Current.K.clear_session();
        }
    }
}
