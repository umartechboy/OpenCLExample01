using Cloo;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenCL_2;
using System.Reflection;
using OpenCL.NetCore;
using ConsoleTables;
using OpenCl.DotNetCore.Memory;
using OpenCl.DotNetCore;
using OpenCL.NetCore.Extensions;

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
            Console.WriteLine("Running the benchmark");

            // 2 million
            var limit = 20_000_000;
            await GetPrimesWithGPU(limit);

            var numbers = Enumerable.Range(0, limit).ToList();
            var watch = Stopwatch.StartNew();
            var primeNumbersFromForeach = GetPrimeList(numbers);
            watch.Stop();

            var watchForParallel = Stopwatch.StartNew();
            var primeNumbersFromParallelForeach = GetPrimeListWithParallel(numbers);
            watchForParallel.Stop();


            Console.WriteLine($"Classical foreach loop | Total prime numbers : {primeNumbersFromForeach.Count} | Time Taken : {watch.ElapsedMilliseconds} ms.");
            Console.WriteLine($"CPU Cores: {System.Environment.ProcessorCount} | Expected time in multi-thread: {watch.ElapsedMilliseconds / System.Environment.ProcessorCount}");
            Console.WriteLine($"Parallel.ForEach loop  | Total prime numbers : {primeNumbersFromParallelForeach.Count} | Time Taken : {watchForParallel.ElapsedMilliseconds} ms.");


            Console.WriteLine("Press any key to exit.");
            Console.ReadLine();
        }

        async static void OpenCLCode3()
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

            // Gets the first available platform and selects the first device offered by the platform and prints out the chosen device
            var chosenDevice = platforms?.FirstOrDefault()?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.All).FirstOrDefault();
            Console.WriteLine($"Using: {chosenDevice?.Name} ({chosenDevice?.Vendor})");
            Console.WriteLine();

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
        }

        static async Task OpenCLSquaresSample()
        {
            Console.WriteLine("NetCore Wrapper (squares)");
            // Gets all available platforms and their corresponding devices, and prints them out in a table
            IEnumerable<OpenCl.DotNetCore.Platforms.Platform> platforms = OpenCl.DotNetCore.Platforms.Platform.GetPlatforms();
            
            // Gets the first available platform and selects the first device offered by the platform and prints out the chosen device
            var chosenDevice = platforms?.FirstOrDefault()?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.All).FirstOrDefault();
            Console.WriteLine($"Using: {chosenDevice?.Name} ({chosenDevice?.Vendor})");
            Console.WriteLine();

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
                        var source = Enumerable.Range(0, 200).Select(v => (float)v).ToArray();
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
        }

        static async Task<List<int>> GetPrimesWithGPU(int MaxNum)
        {
            var totalSW = Stopwatch.StartNew();
            // Gets all available platforms and their corresponding devices, and prints them out in a table
            IEnumerable<OpenCl.DotNetCore.Platforms.Platform> platforms = OpenCl.DotNetCore.Platforms.Platform.GetPlatforms();

            // Gets the first available platform and selects the first device offered by the platform and prints out the chosen device
            var chosenDevice = platforms?.FirstOrDefault()?.GetDevices(OpenCl.DotNetCore.Devices.DeviceType.All).FirstOrDefault();

            var sw = Stopwatch.StartNew();
            // Creats a new context for the selected device
            using (var context = OpenCl.DotNetCore.Contexts.Context.CreateContext(chosenDevice))
            {
                // Creates the kernel code, which multiplies a matrix with a vector
                string code = @"
                    __kernel void square(__global int* result) {
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
                    using (var kernel = program.CreateKernel("square"))
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

        static void OpenCLCode()
        {
            Console.WriteLine("AddArray using OpenCL");
            //Generate the arrays for the demo
            const int arrayLength = 20;
            int[] a = new int[arrayLength];
            int[] b = new int[arrayLength];
            int[] c = new int[arrayLength]; //this will be the array storing the sum result
            for (int i = 0; i < arrayLength; i++)
            {
                a[i] = i;
                b[i] = i;
            }

            //initialize the OpenCL object
            OpenCL_2.OpenCL cl = new OpenCL_2.OpenCL();
            cl.Accelerator = AcceleratorDevice.GPU;

            //Load the kernel from the file into a string
            string kernel = @"
__kernel void addArray(__global int* a, __global int* b, __global int* c)
{
    int id = get_global_id(0);
    c[id] = a[id] + b[id];
printf(""%d + %d = %d\r\n"", a[id], b[id], c[id]);
}
";

            //Specify the kernel code (in our case in the variable kernel) and which function from the kernel file we intend to call
            cl.SetKernel(kernel, "addArray");

            //Specify the parameters in the same order as in the function definition
            cl.SetParameter(a, b, c);

            /*
             *Specify the number of worker threads, 
             * in our case the same as the number of elements since every threads processes only one element,
             * and launch the code on the GPU
             */
            cl.Execute(arrayLength);


            Console.WriteLine("Done...");
            for (int i = 0; i < arrayLength; i++)
            {
                Console.WriteLine($"{a[i]} + {b[i]} = {c[i]}"); //print the result on screen
            }
            Console.ReadKey();
        }
        static string IsPrimeHW
        {
            get
            {
                return @"
        kernel void GetIfPrime(global int* message)
        {
            int index = get_global_id(0);

            int upperl=(int)sqrt((float)message[index]);
            for(int i=2;i<=upperl;i++)
            {
                if(message[index]%i==0)
                {
                    //printf("" %d / %d\n"",index,i );
                    message[index]=0;
                    return;
                }
            }
            printf("" % d"",index);
        }";
            }
        }
        static List<int> GetPrimeListWithHW_Obsolete(AcceleratorDevice Device, int upper)
        {
            Console.Write($"Compiling Kernel | ");
            var watch = Stopwatch.StartNew();
            ComputeMethod method = new ComputeMethod(IsPrimeHW, "GetIfPrime", Device);
            watch.Stop();
            Console.WriteLine($"Took: {watch.ElapsedMilliseconds}ms");
            int[] values = Enumerable.Range(0, upper).ToArray();
            watch.Restart();
            method.Invoke(values.Length, values);
            
            watch.Stop();
            var primes = values.Where(n => n != 0).ToList();
            Console.WriteLine(Device);
            Console.WriteLine($"GPU worker | Total prime numbers : {primes.Count} | Time Taken : {watch.ElapsedMilliseconds} ms.");

            return primes;
        }
        /// <summary>
        /// GetPrimeList returns Prime numbers by using sequential ForEach
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        private static IList<int> GetPrimeList(IList<int> numbers) => numbers.Where(IsPrime).ToList();

        /// <summary>
        /// GetPrimeListWithParallel returns Prime numbers by using Parallel.ForEach
        /// </summary>
        /// <param name="numbers"></param>
        /// <returns></returns>
        private static IList<int> GetPrimeListWithParallel(IList<int> numbers)
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