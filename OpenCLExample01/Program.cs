using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Reflection;
using ConsoleTables;
using OpenCl.DotNetCore.Memory;
using OpenCl.DotNetCore;

namespace OpenCLExample01
{
    internal class Program
    {
        static void Main()
        {
            MainAsync().Wait();
        }
        async static Task MainAsync()
        {
            ShowOpenCLInfo();
            var platform = OpenCl.DotNetCore.Platforms.Platform.GetPlatforms()?.FirstOrDefault();

            Console.WriteLine("Running sample codes");
            await Example_OpenCLMatrixVectorMultiplication(platform?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.Gpu).FirstOrDefault());
            await Example_OpenCLSquaresSample(platform?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.Gpu).FirstOrDefault());

            // List of functions is available at https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#built-in-functions
            // Lisy of Operators is available at https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#operators
            var addition = await OpenCLBinaryOperator(new float[] { 1, 2, 3, 4, 5, 6 }, new float[] { 100, 100, 100, 100, 100, 100 }, "+");
            Console.WriteLine($"Adder: {string.Join(", ", addition)}"); AddSpacer();
            var multiplication = await OpenCLBinaryOperator(new float[] { 1, 2, 3, 4, 5, 6 }, new float[] { 100, 100, 100, 100, 100, 100 }, "*");
            Console.WriteLine($"Multiplier: {string.Join(", ", multiplication)}"); AddSpacer();

            // Kernel from Project File kernel.c can be fetched using System.IO.File.ReadAllText("kernel.c");

            Console.WriteLine("Running the prime number tests");

            // 2 million
            var limit = 20_000_000;
            await GetPrimesWithOpenCL(limit, platform?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.Gpu).FirstOrDefault());
            //await GetPrimesWithOpenCL(limit, platform?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.Cpu).FirstOrDefault());
            //await GetPrimesWithOpenCL(limit, platform?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.All).FirstOrDefault());
            AddSpacer();

            var numbers = Enumerable.Range(0, limit).ToList();
            var watch = Stopwatch.StartNew();
            var primeNumbersFromForeach = GetPrimeSingleCore(numbers);
            watch.Stop();

            var watchForParallel = Stopwatch.StartNew();
            var primeNumbersFromParallelForeach = GetPrimesWithMultiThreads(numbers);
            watchForParallel.Stop();

            Console.WriteLine($"Classical foreach loop | Total prime numbers : {primeNumbersFromForeach.Count} | Time Taken : {watch.ElapsedMilliseconds} ms.");
            Console.WriteLine($"CPU Cores: {System.Environment.ProcessorCount} | Expected time in multi-thread: {watch.ElapsedMilliseconds / System.Environment.ProcessorCount}");
            Console.WriteLine($"Parallel.ForEach loop  | Total prime numbers : {primeNumbersFromParallelForeach.Count} | Time Taken : {watchForParallel.ElapsedMilliseconds} ms.");


            Console.WriteLine("Press any key to exit.");
            Console.ReadLine();
        }
        static void AddSpacer()
        {
            Console.WriteLine("\r\n---------------------------------------------\r\n");
        }
        static void ShowOpenCLInfo()
        {
            // Gets all available platforms and their corresponding devices, and prints them out in a table
            IEnumerable<OpenCl.DotNetCore.Platforms.Platform> platforms = OpenCl.DotNetCore.Platforms.Platform.GetPlatforms();
            ConsoleTable consoleTable = new ConsoleTable("Platform", "OpenCL Version", "Vendor", "Device", "Driver Version", "Bits", "Memory", "Clock Speed", "Available");
            foreach (var platform in platforms)
            {
                foreach (var device in platform.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.All))
                {
                    consoleTable.AddRow(
                        platform.Name,
                        $"{platform.Version.MajorVersion}.{platform.Version.MinorVersion}",
                        platform.Vendor,
                        device.Name,
                        device.DriverVersion,
                        $"{device.AddressBits} Bit",
                        $"{Math.Round(device.GlobalMemorySize / 1024.0f / 1024.0f / 1024.0f, 2)} GiB",
                        $"{device.MaximumClockFrequency} MHz",
                        device.IsAvailable ? "✔" : "✖");
                }
            }
            Console.WriteLine("Supported Platforms & Devices:");
            consoleTable.Write(Format.Alternative);
        }

        static async Task<float[]> OpenCLBinaryOperator(float[] Array1, float[] Array2, string operator_)
        {
            var platform = OpenCl.DotNetCore.Platforms.Platform.GetPlatforms()?.FirstOrDefault();
            var chosenDevice = platform?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.Gpu).FirstOrDefault();

            // Creats a new context for the selected device
            using (var context = OpenCl.DotNetCore.Contexts.Context.CreateContext(chosenDevice))
            {
                // Creates the kernel code, which multiplies a matrix with a vector
                string code = @"
                    __kernel void BinOp(__global float* a, __global float* b, __global float* res) {
                        int i = get_global_id(0);
                        res[i] = a[i] " + operator_ + @" b[i];
                    }";

                // Creates a program and then the kernel from it
                using (var program = await context.CreateAndBuildProgramFromStringAsync(code))
                {
                    using (var kernel = program.CreateKernel("BinOp"))
                    {
                        // Creates the memory objects for the input arguments of the kernel
                        MemoryBuffer source1 = context.CreateBuffer(MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, Array1);
                        MemoryBuffer source2 = context.CreateBuffer(MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, Array2);
                        MemoryBuffer resultBuffer = context.CreateBuffer<float>(MemoryFlag.WriteOnly, Array1.Length); // lengths must match

                        // Tries to execute the kernel
                        try
                        {
                            // Sets the arguments of the kernel
                            kernel.SetKernelArgument(0, source1);
                            kernel.SetKernelArgument(1, source2);
                            kernel.SetKernelArgument(2, resultBuffer);
                            //kernel.SetKernelArgument(1, resultBuffer);

                            // Creates a command queue, executes the kernel, and retrieves the result
                            using (var commandQueue = OpenCl.DotNetCore.CommandQueues.CommandQueue.CreateCommandQueue(context, chosenDevice))
                            {
                                commandQueue.EnqueueNDRangeKernel(kernel, 1, Array1.Length); ;
                                float[] resultArray = await commandQueue.EnqueueReadBufferAsync<float>(resultBuffer, Array1.Length);

                                source1.Dispose();
                                source2.Dispose();
                                resultBuffer.Dispose();
                                return resultArray;
                            }
                        }
                        catch (OpenClException exception)
                        {
                            Console.WriteLine(exception.Message);
                        }

                        // Disposes of the memory objects
                        source1.Dispose();
                        source2.Dispose();
                        resultBuffer.Dispose();
                    }
                }
            }
            return null;
        }


        async static Task Example_OpenCLMatrixVectorMultiplication(OpenCl.DotNetCore.Devices.Device chosenDevice)
        {
            Console.WriteLine("4x4 Matrix Multiplied with Vector 4. Device: " + chosenDevice.Name);

            // Creats a new context for the selected device
            using (var context = OpenCl.DotNetCore.Contexts.Context.CreateContext(chosenDevice))
            {
                // Creates the kernel code, which multiplies a matrix with a vector
                string code = @"
                    __kernel void matvec_mult(__global float4* matrix,
                                              __global float4* vector,
                                              __global float* result) {
                        int i = get_global_id(0);
                        result[i] = dot(matrix[i], vector[0]);
                        printf(""in kernel: %d\r\n"", i);
                    }";

                // Creates a program and then the kernel from it
                using (var program = await context.CreateAndBuildProgramFromStringAsync(code))
                {
                    using (var kernel = program.CreateKernel("matvec_mult"))
                    {
                        // Creates the memory objects for the input arguments of the kernel
                        MemoryBuffer matrixBuffer = context.CreateBuffer(MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new float[]
                        {
                             2f,  0f,  0f,  0f,
                             0f, 2f, 0f, 0f,
                            0f, 0f, 2f, 0f,
                            0f, 0f, 0f, 2f
                        });
                        MemoryBuffer vectorBuffer = context.CreateBuffer(MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, new float[]
                        { 0f, 3f, 6f, 9f });
                        MemoryBuffer resultBuffer = context.CreateBuffer<float>(MemoryFlag.WriteOnly, 4);

                        // Tries to execute the kernel
                        try
                        {
                            // Sets the arguments of the kernel
                            kernel.SetKernelArgument(0, matrixBuffer);
                            kernel.SetKernelArgument(1, vectorBuffer);
                            kernel.SetKernelArgument(2, resultBuffer);

                            // Creates a command queue, executes the kernel, and retrieves the result
                            using (var commandQueue = OpenCl.DotNetCore.CommandQueues.CommandQueue.CreateCommandQueue(context, chosenDevice))
                            {
                                commandQueue.EnqueueNDRangeKernel(kernel, 1, 4);
                                float[] resultArray = await commandQueue.EnqueueReadBufferAsync<float>(resultBuffer, 4);
                                Console.WriteLine($"Result: ({string.Join(", ", resultArray)})");
                            }
                        }
                        catch (OpenClException exception)
                        {
                            Console.WriteLine(exception.Message);
                        }

                        // Disposes of the memory objects
                        matrixBuffer.Dispose();
                        vectorBuffer.Dispose();
                        resultBuffer.Dispose();
                    }
                }
            }
            AddSpacer();
        }

        static async Task Example_OpenCLSquaresSample(OpenCl.DotNetCore.Devices.Device chosenDevice)
        {
            Console.WriteLine("Squares with OpenCL. Device: " + chosenDevice.Name);

            // Creats a new context for the selected device
            using (var context = OpenCl.DotNetCore.Contexts.Context.CreateContext(chosenDevice))
            {
                // Creates the kernel code, which multiplies a matrix with a vector
                string code = @"
                    __kernel void square(__global float* given) {
                        int i = get_global_id(0);
                        given[i] = given[i] * given[i];
                        //printf(""in kernel: %d\r\n"", i);
                    }";

                // Creates a program and then the kernel from it
                using (var program = await context.CreateAndBuildProgramFromStringAsync(code))
                {
                    using (var kernel = program.CreateKernel("square"))
                    {
                        // Creates the memory objects for the input arguments of the kernel
                        var source = Enumerable.Range(0, 20).Select(v => (float)v).ToArray();
                        MemoryBuffer given = context.CreateBuffer(MemoryFlag.ReadOnly | MemoryFlag.CopyHostPointer, source);
                        MemoryBuffer resultBuffer = context.CreateBuffer<float>(MemoryFlag.WriteOnly, source.Length);

                        // Tries to execute the kernel
                        try
                        {
                            // Sets the arguments of the kernel
                            kernel.SetKernelArgument(0, given);
                            //kernel.SetKernelArgument(1, resultBuffer);

                            // Creates a command queue, executes the kernel, and retrieves the result
                            using (var commandQueue = OpenCl.DotNetCore.CommandQueues.CommandQueue.CreateCommandQueue(context, chosenDevice))
                            {
                                commandQueue.EnqueueNDRangeKernel(kernel, 1, source.Length); ;
                                float[] resultArray = await commandQueue.EnqueueReadBufferAsync<float>(given, source.Length);
                                Console.WriteLine($"Result: ({string.Join(", ", resultArray)})");
                            }
                        }
                        catch (OpenClException exception)
                        {
                            Console.WriteLine(exception.Message);
                        }

                        // Disposes of the memory objects
                        given.Dispose();
                        resultBuffer.Dispose();
                    }
                }
            }
            AddSpacer();
        }

        static async Task<List<int>> GetPrimesWithOpenCL(int MaxNum, OpenCl.DotNetCore.Devices.Device chosenDevice)
        {
            Console.WriteLine("GetPrimes with OpenCL. Device: " + chosenDevice.Name);
            var totalSW = Stopwatch.StartNew();

            var sw = Stopwatch.StartNew();
            // Creats a new context for the selected device
            using (var context = OpenCl.DotNetCore.Contexts.Context.CreateContext(chosenDevice))
            {
                // Creates the kernel code, which multiplies a matrix with a vector
                string code = @"
                    __kernel void getIfPrime(__global int* result) {
                        int index = get_global_id(0) + 2;

                        int upperl=(int)sqrt((float)index);
                        for(int i=2;i<=upperl;i++)
                        {
                            if(index%i==0)
                            {
                                //printf("" %d / %d\n"",index,i );
                                result[index]=0;
                                return;
                            }
                        }

                        result[index - 2]=index;
                        //printf("" % d"",index);
                    }";

                // Creates a program and then the kernel from it
                using (var program = await context.CreateAndBuildProgramFromStringAsync(code))
                {
                    using (var kernel = program.CreateKernel("getIfPrime"))
                    {
                        Console.WriteLine($"OpenCL | Kernel creation: {sw.ElapsedMilliseconds}ms.");
                        sw.Restart();
                        // Creates the memory objects for the input arguments of the kernel
                        MemoryBuffer answersB = context.CreateBuffer<int>(MemoryFlag.ReadWrite, MaxNum);

                        Console.WriteLine($"OpenCL | Argument transfer: {sw.ElapsedMilliseconds}ms.");
                        sw.Restart();
                        // Tries to execute the kernel
                        try
                        {
                            // Sets the arguments of the kernel
                            kernel.SetKernelArgument(0, answersB);
                            //kernel.SetKernelArgument(1, resultBuffer);

                            // Creates a command queue, executes the kernel, and retrieves the result
                            using (var commandQueue = OpenCl.DotNetCore.CommandQueues.CommandQueue.CreateCommandQueue(context, chosenDevice))
                            {
                                commandQueue.EnqueueNDRangeKernel(kernel, 1, MaxNum);
                                int[] resultArray = await commandQueue.EnqueueReadBufferAsync<int>(answersB, MaxNum);

                                Console.WriteLine($"OpenCL | Execution: {sw.ElapsedMilliseconds}ms.");
                                Console.WriteLine($"OpenCL | Total Primes: {resultArray.ToList().Count(v => v != 0)}");
                                Console.WriteLine($"OpenCL | Total Time: {totalSW.ElapsedMilliseconds}ms.");
                                answersB.Dispose();

                                return resultArray.ToList();
                                //Console.WriteLine($"Result: ({string.Join(", ", resultArray)})");
                            }
                        }
                        catch (OpenClException exception)
                        {
                            Console.WriteLine(exception.Message);
                        }

                        // Disposes of the memory objects
                        answersB.Dispose();
                    }
                }
            }
            return null;
        }
        /// <summary>
        /// GetPrimeList returns Prime numbers by using sequential ForEach The least economical way of doing things around here
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        private static IList<int> GetPrimeSingleCore(IList<int> numbers)
        {
            return numbers.Where(IsPrime).ToList();
        }

        /// <summary>
        /// GetPrimeListWithParallel returns Prime numbers by using Parallel.ForEach. Uses maximum CPU at least
        /// </summary>
        /// <param name="numbers"></param>
        /// <returns></returns>
        private static IList<int> GetPrimesWithMultiThreads(IList<int> numbers)
        {
            var primeNumbers = new ConcurrentBag<int>();

            Parallel.ForEach(numbers, (number) =>
            {
                if (IsPrime(number))
                {
                    primeNumbers.Add(number);
                }
            });

            return primeNumbers.ToList();
        }

        /// <summary>
        /// IsPrime returns true if number is Prime, else false.(https://en.wikipedia.org/wiki/Prime_number)
        /// </summary>
        /// <param name="number"></param>
        /// <returns></returns>
        private static bool IsPrime(int number)
        {
            if (number < 2)
            {
                return false;
            }

            for (var divisor = 2; divisor <= Math.Sqrt(number); divisor++)
            {
                if (number % divisor == 0)
                {
                    return false;
                }
            }
            return true;
        }
    }
}