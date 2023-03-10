﻿using Cloo;
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
            Console.WriteLine("Running the benchmark");

            OpenCLCode3();
            Console.ReadKey();
            return;
            // 2 million
            var limit = 2_00;
            var numbers = Enumerable.Range(0, limit).ToList();

            var watch = Stopwatch.StartNew();
            var primeNumbersFromForeach = GetPrimeList(numbers);
            watch.Stop();

            var watchForParallel = Stopwatch.StartNew();
            var primeNumbersFromParallelForeach = GetPrimeListWithParallel(numbers);
            watchForParallel.Stop();

            foreach (var AcceleratorDevice in AcceleratorDevice.All)
            {
                GetPrimeListWithHW(AcceleratorDevice, limit);
            }


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
        static void OpenCLCode2() {

            // Initialize OpenCL
            
            Platform[] platforms = Cl.GetPlatformIDs(out ErrorCode er1);
            Device[] devices = Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out ErrorCode er2);
            Context context = Cl.CreateContext(null, 1, devices, null, IntPtr.Zero, out ErrorCode er3);
            CommandQueue commandQueue = Cl.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, out ErrorCode er4);

            // Create kernel program
            string kernelSource = @"
                __kernel void square(__global float* array)
                {
                    int index = get_global_id(0);
                    array[index] = array[index] * array[index];
                    printf(""Ind %d"", index);
                }";
            var program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out ErrorCode er5);
            Cl.BuildProgram(program, 1, devices, null, null, IntPtr.Zero);

            // Create kernel
            Kernel kernel = Cl.CreateKernel(program, "square", out ErrorCode er6);

            // Create and populate array
            const int arraySize = 10;
            float[] array = new float[arraySize];
            for (int i = 0; i < arraySize; i++)
            {
                array[i] = i;
            }

            // Create memory buffer
            var arrayBuffer = Cl.CreateBuffer(context, MemFlags.ReadOnly, arraySize * sizeof(float), out ErrorCode er7);

            // Write array to buffer
            Cl.EnqueueWriteBuffer(commandQueue, arrayBuffer, Bool.True, IntPtr.Zero, (IntPtr)(arraySize * sizeof(float)), array, 0, null, out Event writeEvent);

            // Set kernel arguments
            Cl.SetKernelArg(kernel, 0, (IntPtr)sizeof(int), arrayBuffer);

            // Execute kernel
            IntPtr[] workGroupSize = { (IntPtr)arraySize };
            Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, new IntPtr[] { IntPtr.Zero }, workGroupSize, null, 0, null, out Event kernelEvent);

            // Wait for kernel to finish executing
            Cl.Finish(commandQueue);

            // Read modified array from buffer
            Cl.EnqueueReadBuffer(commandQueue, arrayBuffer, Bool.True, IntPtr.Zero, (IntPtr)(arraySize * sizeof(float)), array, 0, null, out Event readEvent);

            // Print modified array
            for (int i = 0; i < arraySize; i++)
            {
                Console.WriteLine(array[i]);
            }

            // Clean up resources
            Cl.ReleaseKernel(kernel);
            Cl.ReleaseProgram(program);
            Cl.ReleaseMemObject(arrayBuffer);
            Cl.ReleaseCommandQueue(commandQueue);
            Cl.ReleaseContext(context);
        }
        static void OpenCLCode()
        {
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
__kernel void addArray(global int* a, global int* b, global int* c)
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
        static List<int> GetPrimeListWithHW(AcceleratorDevice Device, int upper)
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