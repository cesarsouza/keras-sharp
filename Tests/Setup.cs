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
    public class MySetUpClass
    {
        [OneTimeSetUp]

        public void RunBeforeAnyTests()
        {
            // This setup is necessary to be able to run CNTK tests
            Directory.SetCurrentDirectory(Path.GetDirectoryName(typeof(MySetUpClass).Assembly.Location));
        }
    }
}
