using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests
{
    public static class AssertEx
    {
        public static void AreEqual(dynamic a, dynamic b, dynamic rtol = null)
        {
            if (rtol == null)
                Assert.IsTrue(Accord.Math.Matrix.IsEqual(a, b));
            else
                Assert.IsTrue(Accord.Math.Matrix.IsEqual(a, b, rtol: rtol));
        }
    }
}
