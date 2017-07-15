using KerasSharp;
using KerasSharp.Models;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

            Assert.Fail("This is just an example - the test has still to be written");
        }
    }
}
