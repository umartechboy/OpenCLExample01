using Cloo;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenCL;
using System.Reflection;

namespace OpenCLExample01
{
    internal class Program
    {
        static void Main()
        {
            Console.WriteLine("Running the benchmark");

            OpenCLCode();
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
            OpenCL.OpenCL cl = new OpenCL.OpenCL();
            cl.Accelerator = AcceleratorDevice.GPU;

            //Load the kernel from the file into a string
            string kernel = @"
kernel void addArray(global int* a, global int* b, global int* c)
{
    int id = get_global_id(0);
    c[id] = a[id] + b[id];
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
        static ComputeContext context;
        static ComputeCommandQueue queue;
        static ComputeProgram program;

        static string LastMethod = null;
        static ComputeKernel LastKernel = null;
        static string kernel;
        static AcceleratorDevice _device;
        public static AcceleratorDevice Accelerator
        {
            get
            {
                return _device;
            }
            set
            {
                if (value != _device)
                {
                    _device = value;
                    CreateContext();
                    if (kernel != null)
                    {
                        LoadKernel(kernel);
                    }
                }
            }
        }

        static void CreateContext()
        {
            context = new ComputeContext(_device.Type, new ComputeContextPropertyList(Accelerator.Device.Platform), null, IntPtr.Zero);
            queue = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.None);
        }
        public static void LoadKernel(string Kernel)
        {
            kernel = Kernel;
            program = new ComputeProgram(context, Kernel);

            try
            {
                program.Build(null, null, null, IntPtr.Zero);   //compile
            }
            catch (BuildProgramFailureComputeException)
            {
                string message = program.GetBuildLog(Accelerator.Device);
                throw new ArgumentException(message);
            }
        }
        static ComputeKernel CreateKernel(string Method, object[] args)
        {
            if (args == null) throw new ArgumentException("You have to pass an argument to a kernel");

            ComputeKernel kernel;
            if (LastMethod == Method && LastKernel != null) //Kernel caching, do not compile twice
            {
                kernel = LastKernel;
            }
            else
            {
                kernel = program.CreateKernel(Method);
                LastKernel = kernel;
            }
            LastMethod = Method;

            for (int i = 0; i < args.Length; i++)
            {
                Setargument(kernel, i, args[i]);
            }

            return kernel;
        }

        static void Setargument(ComputeKernel kernel, int index, object arg)
        {
            if (arg == null) throw new ArgumentException("Argument " + index + " is null");

            Type argtype = arg.GetType();
            if (argtype.IsArray)
            {
                ComputeMemory? messageBuffer = Activator.CreateInstance(typeof(ComputeBuffer<>).MakeGenericType(argtype.GetElementType()), new object[]
                {
                    context,
                    ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer,
                    arg
                }) as ComputeMemory;
                kernel.SetMemoryArgument(index, messageBuffer); // set the array
            }
            else
            {
                typeof(ComputeKernel).GetMethod("SetValueArgument")?.MakeGenericMethod(argtype).Invoke(kernel, new object[] { index, arg });
            }
        }
    }
}