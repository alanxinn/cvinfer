#if NET472
namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit
    {
    }
}

namespace System
{
    internal static class HashCode
    {
        public static int Combine<T1>(T1 value1) =>
            System.Collections.Generic.EqualityComparer<T1>.Default.GetHashCode(value1);

        public static int Combine<T1, T2>(T1 value1, T2 value2) =>
            CombineCore(value1, value2);

        public static int Combine<T1, T2, T3>(T1 value1, T2 value2, T3 value3) =>
            CombineCore(value1, value2, value3);

        public static int Combine<T1, T2, T3, T4>(T1 value1, T2 value2, T3 value3, T4 value4) =>
            CombineCore(value1, value2, value3, value4);

        public static int Combine<T1, T2, T3, T4, T5>(T1 value1, T2 value2, T3 value3, T4 value4, T5 value5) =>
            CombineCore(value1, value2, value3, value4, value5);

        public static int Combine<T1, T2, T3, T4, T5, T6>(
            T1 value1, T2 value2, T3 value3, T4 value4, T5 value5, T6 value6) =>
            CombineCore(value1, value2, value3, value4, value5, value6);

        private static int CombineCore(params object?[] values)
        {
            unchecked
            {
                var hash = 17;
                foreach (var value in values)
                    hash = hash * 31 + (value?.GetHashCode() ?? 0);
                return hash;
            }
        }
    }
}
#endif
