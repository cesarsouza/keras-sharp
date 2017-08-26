// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

namespace KerasSharp
{
    using Accord.Statistics.Kernels;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Metrics;
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;

    public static class Python
    {
        // Mappings from Python calls to .NET
        static ObjectIDGenerator generator = new ObjectIDGenerator();

        public static int[] range(int a, int b)
        {
            return Accord.Math.Vector.Range(a, b);
        }

        public static int[] range(int? a, int? b)
        {
            return range(a.Value, b.Value);
        }

        public static long id(object x)
        {
            if (x == null)
                return 0;

            bool firstTime;
            return generator.GetId(x, out firstTime);
        }

        public static int?[] tuple(params int?[] values)
        {
            return values;
        }

        public static string str(object obj)
        {
            if (obj == null)
                return "null";

            if (obj is IEnumerable)
            {
                var l = new List<string>();
                foreach (object o in (IEnumerable)obj)
                    l.Add(str(o));

                return "[" + String.Join(", ", l.ToArray()) + "]";
            }
            else if (obj is IDictionary)
            {
                var dict = obj as IDictionary;
                var l = new List<string>();
                foreach (object k in dict.Keys)
                    l.Add($"{str(k)}: {str(dict[k])}");

                return "{" + String.Join(", ", l.ToArray()) + "}";
            }

            return obj.ToString();
        }

        public static TValue get<TKey, TValue>(this IDictionary<TKey, TValue> dict, TKey key, TValue def)
        {
            TValue r;
            if (dict.TryGetValue(key, out r))
                return r;
            return def;
        }

        public static int len(this ICollection list)
        {
            return list.Count;
        }

        public static bool hasattr(object obj, string name)
        {
            throw new NotImplementedException();
        }

        public static T getattr<T>(object layer, object attr, object[] v)
        {
            throw new NotImplementedException();
        }


        // Methods to condense single elements and lists into dictionary
        // so they can be passed more easily along methods that follow
        // the Python interfaces. We include some markings to be able
        // to detect what those values originally were before being
        // transformed to dictionaries.

        public static Dictionary<string, T> dict_from_single<T>(this T value)
        {
            return new Dictionary<string, T>() { { "__K__single__", value } };
        }

        public static Dictionary<string, T> dict_from_list<T>(this List<T> list)
        {
            var dict = new Dictionary<string, T>();
            for (int i = 0; i < list.Count; i++)
                dict["__K__list__" + i] = list[i];
            return dict;
        }

        public static bool is_dict<T>(this Dictionary<string, T> dict)
        {
            if (dict == null)
                return false;
            return !dict.Keys.Any(x => x.StartsWith("__K__"));
        }

        public static bool is_list<T>(this Dictionary<string, T> dict)
        {
            if (dict == null)
                return false;
            return dict.Keys.All(x => x.StartsWith("__K__list__"));
        }

        public static bool is_single<T>(this Dictionary<string, T> dict)
        {
            if (dict == null)
                return true;
            return dict.Keys.Count == 1 && dict.ContainsKey("__K__single__");
        }

        public static List<T> to_list<T>(this Dictionary<string, T> dict)
        {
            List<T> list = new List<T>();
            for (int i = 0; i < dict.Keys.Count; i++)
                list.Add(dict["__K__list__" + i]);
            return list;
        }

        public static T to_single<T>(this Dictionary<string, T> dict)
        {
            if (dict == null)
                return default(T);
            return dict["__K__single__"];
        }



        public static List<T> Get<T>(IList<string> name)
        {
            if (name == null)
                return null;
            return name.Select(s => Get<T>(s)).ToList();
        }

        public static T Get<T>(string name)
        {
            Type baseType = typeof(T);
            Type foundType = get(name, baseType);

            if (foundType == null)
                foundType = get(name.Replace("_", ""), baseType); // try again after normalizing the name

            if (foundType == null)
                throw new ArgumentOutOfRangeException("name", $"Could not find {baseType.Name} '{name}'.");

            return (T)Activator.CreateInstance(foundType);
        }

        private static Type get(string name, Type type)
        {
            return AppDomain.CurrentDomain.GetAssemblies()
                                        .SelectMany(s => s.GetTypes())
                                        .Where(p => type.IsAssignableFrom(p) && !p.IsInterface)
                                        .Where(p => p.Name.ToUpperInvariant() == name.ToUpperInvariant())
                                        .FirstOrDefault();
        }

        public static List<T> Get<T>(IEnumerable<object> name)
            where T : class
        {
            return name.Select(s => Get<T>(s)).ToList();
        }

        public static T Get<T>(object obj)
            where T : class
        {
            if (obj is String)
                return Get<T>(obj as string);

            if (typeof(T) == typeof(IMetric))
            {
                var f2 = obj as Func<Tensor, Tensor, Tensor>;
                if (f2 != null)
                    return new CustomMetric(f2) as T;

                var f3 = obj as Func<Tensor, Tensor, Tensor, Tensor>;
                if (f3 != null)
                    return new CustomMetric(f3) as T;
            }

            throw new NotImplementedException();
        }
    }
}
