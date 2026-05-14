using System;
using System.Xml.Linq;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;

static class Program
{
    const int ATTEMPTS = 100;
    const int MAX_OBSERVATIONS = 100000; 

    enum Strategy { Baseline, Fixed, Relative }

    static void Main()
    {
        Directory.CreateDirectory("data");
        XDocument xdoc = XDocument.Load("samples.xml");
        string csvPath = "data/focused_profiling_results.csv";

        using var csv = new StreamWriter(csvPath);
        csv.WriteLine("strategy,parameter,name,filename,N,size,periodic,symmetry,heuristic,attempts,contradictions,successes,contradiction_rate,avg_observations");

        // 1. Run Baseline (0 Backtracks) + timing
        Console.WriteLine("\n==============================================");
        Console.WriteLine("   STARTING BASELINE SIMULATION (0 BACKTRACKS)");
        Console.WriteLine("==============================================");
        var baselineStopwatch = Stopwatch.StartNew();
        RunSimulation(xdoc, csv, Strategy.Baseline, 0);
        baselineStopwatch.Stop();
        Console.WriteLine($"Baseline (0 backtracks) total time: {baselineStopwatch.Elapsed}");

        // 2. Test Fixed Backtracks limit distances
        //for (int backtrackLimit = 100; backtrackLimit <= 2000; backtrackLimit += 100)
        for (int backtrackLimit = 250; backtrackLimit <= 850; backtrackLimit += 25)
        {
            Console.WriteLine("\n==============================================");
            Console.WriteLine($"   STARTING FIXED BACKTRACK (LIMIT {backtrackLimit})");
            Console.WriteLine("==============================================");
            RunSimulation(xdoc, csv, Strategy.Fixed, backtrackLimit);
        }

        //// 3. Test Progress-Relative Backtracks
        //double[] relativeFactors = { 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,  0.70,  0.80,  0.90 };
        //foreach (var factor in relativeFactors)
        //{
        //    Console.WriteLine("\n==============================================");
        //    Console.WriteLine($"   STARTING RELATIVE BACKTRACK (FACTOR {factor})");
        //    Console.WriteLine("==============================================");
        //    RunSimulation(xdoc, csv, Strategy.Relative, factor);
        //}
        
        Console.WriteLine($"\n simulations complete - combined CSV saved to {Path.GetFullPath(csvPath)}");
    }

    static void RunSimulation(XDocument xdoc, StreamWriter csv, Strategy strategy, float strategyParam)
    {
        Random random = new(42); 
        var results = new List<(string label, int contradictions, int successes)>();

        foreach (XElement xelem in xdoc.Root.Elements("overlapping"))
        {
            string name = xelem.Get<string>("name");
            int N = xelem.Get("N", 3);
            bool periodic = xelem.Get("periodic", false);
            int size = xelem.Get("size", 48);

            // Skip exactly: "Chess N=2 size=47 periodic=True"
            if (string.Equals(name?.Trim(), "Chess", StringComparison.OrdinalIgnoreCase)
                && N == 2
                && size == 47
                && periodic)
            {
                Console.WriteLine("Skipping Chess N=2 size=47 periodic=True");
                continue;
            }

            int width = xelem.Get("width", size);
            int height = xelem.Get("height", size);
            bool periodicInput = xelem.Get("periodicInput", true);
            int symmetry = xelem.Get("symmetry", 8);
            bool ground = xelem.Get("ground", false);
            string heuristicString = xelem.Get<string>("heuristic");
            var heuristic = heuristicString == "Scanline" ? Model.Heuristic.Scanline :
                            (heuristicString == "MRV" ? Model.Heuristic.MRV : Model.Heuristic.Entropy);

            string label = $"{name} N={N} size={size} periodic={periodic}";
            
            var model = new OverlappingModel(name, N, width, height, periodicInput, periodic, symmetry, ground, heuristic);

            int contradictions = 0;
            int successes = 0;
            bool imageSaved = false;
            string imageFilename = $"{name}_N{N}_size{size}.png";

            long totalObservations = 0;

            Console.Write($"Profiling {label}... ");

            Func<int, int> getDivergencePoint = strategy switch
            {
                Strategy.Baseline => (depth) => -1,
                Strategy.Fixed => (depth) => Math.Max(0, depth - (int)strategyParam),
                Strategy.Relative => (depth) => Math.Max(0, depth - Math.Max(10, (int)(depth * strategyParam))),
                _ => (depth) => -1
            };

            for (int k = 0; k < ATTEMPTS; k++)
            {
                int seed = random.Next();

                bool success = model.Run(seed, xelem.Get("limit", -1), getDivergencePoint, MAX_OBSERVATIONS);

                totalObservations += model.TotalObservations;

                if (success)
                {
                    successes++;
                    if (!imageSaved)
                    {
                        model.Save(Path.Combine("data", imageFilename));
                        imageSaved = true;
                    }
                }
                else
                {
                    contradictions++;
                }
            }

            float rate = MathF.Round((float)contradictions / ATTEMPTS * 100, 1);
            float avgObservations = MathF.Round((float)totalObservations / ATTEMPTS, 1);

            Console.WriteLine($"Avg Obs: {avgObservations} | Success: {successes}/{ATTEMPTS}");

            csv.WriteLine($"{strategy},{strategyParam},{name},{imageFilename},{N},{size},{periodic},{symmetry},{heuristic},{ATTEMPTS},{contradictions},{successes},{rate},{avgObservations}");

            results.Add((label, contradictions, successes));
        }

        Console.WriteLine("\n--- summary - sorted by contradiction rate ---");
        results.Sort((a, b) => b.contradictions.CompareTo(a.contradictions));
        foreach (var (label, contradictions, successes) in results)
        {
            float rate = MathF.Round((float)contradictions / ATTEMPTS * 100, 1);
            string difficulty = rate >= 55 ? "HIGH" : rate >= 15 ? "MEDIUM" : "LOW";
            Console.WriteLine($"[{difficulty,-6}] {rate}% - {label}");
        }
        Console.WriteLine();
    }
}