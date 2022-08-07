
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;

namespace ILGPUTest
{
    class Program
    {
        static void Main()
        {
            using var context = Context.CreateDefault();

            foreach (var device in context.Devices)
            {
                using var accelerator = device.CreateAccelerator(context);
                Console.WriteLine($"Performing operations on {accelerator}");
                var sw = new Stopwatch();
                sw.Restart();
                MyKernel(accelerator);
                sw.Stop();
                Console.WriteLine($"- Naive implementation: {sw.ElapsedMilliseconds}ms");
            }
            Console.ReadLine();
        }
        static void MyCalcute(Index1D total, ArrayView<long> results, Input<int> TotalTag)
        {
            int Total = 0;
            for (int i = 1; i < 10000; i++)
            {
                bool isPrime = true;
                for (int j = 2; j < i - 1; j++)
                    if ((i % j) == 0) {
                        isPrime = false;
                        break;
                    }
                if (isPrime) {
                    results[Total] = i;
                    Total++;
                }
            }
        }
        static void MyKernel(Accelerator accelerator)
        {
            using var buffer = accelerator.Allocate1D<long>(10000);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, Input<int>>(MyCalcute);
            kernel(
                (int)buffer.Length,
                buffer.View,
                10000);

            var results = buffer.GetAsArray1D();
        }
    }
}
