using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests
{
    [SetUpFixture]
    public class Setup
    {
        public const string TENSORFLOW = "KerasSharp.Backends.TensorFlowBackend";
        public const string CNTK = "KerasSharp.Backends.CNTKBackend";


        [OneTimeSetUp]

        public void RunBeforeAnyTests()
        {
            // This setup is necessary to be able to run CNTK tests
            Directory.SetCurrentDirectory(Path.GetDirectoryName(typeof(Setup).Assembly.Location));
        }
    }
}
