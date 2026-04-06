using System;
using System.Xml.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using System.IO;
// dotnet run --configuration Release
static class Program
{
    static void Main()
    {
        const int ATTEMPTS = 1000;

        Directory.CreateDirectory("data");
        Random random = new();
        XDocument xdoc = XDocument.Load("samples.xml");
        string csvPath = "data/profiling_results.csv";
        using var csv = new StreamWriter(csvPath);
        csv.WriteLine("name,filename,N,size,periodic,symmetry,heuristic,attempts,contradictions,successes,contradiction_rate,avg_ms,avg_ms_success,avg_ms_contradiction");
        var results = new List<(string label, int contradictions, int successes)>();

        long totalRuntimeMs = 0;

        foreach (XElement xelem in xdoc.Root.Elements("overlapping"))
        {
            string name = xelem.Get<string>("name");
            int N = xelem.Get("N", 3);
            bool periodic = xelem.Get("periodic", false);
            int size = xelem.Get("size", 48);
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

            long totalMs = 0;
            long totalMsSuccess = 0;
            long totalMsContradiction = 0;

            Console.WriteLine($"Profiling {label}... ");

            for (int k = 0; k < ATTEMPTS; k++)
            {
                int seed = random.Next();

                Stopwatch sw = Stopwatch.StartNew();
                bool success = model.Run(seed, xelem.Get("limit", -1));
                sw.Stop();

                totalMs += sw.ElapsedMilliseconds;

                if (success)
                {
                    successes++;
                    totalMsSuccess += sw.ElapsedMilliseconds;
                    if (!imageSaved)
                    {
                        model.Save(Path.Combine("data", imageFilename));
                        imageSaved = true;
                    }
                }
                else
                {
                    contradictions++;
                    totalMsContradiction += sw.ElapsedMilliseconds;
                }
            }

            double rate = Math.Round((double)contradictions / ATTEMPTS * 100, 1);
            double avgMs = Math.Round((double)totalMs / ATTEMPTS, 2);

            // Avoid division by zero for samples that never contradict or always contradict
            double avgMsSuccess = successes > 0
                ? Math.Round((double)totalMsSuccess / successes, 2)
                : -1;
            double avgMsContradiction = contradictions > 0
                ? Math.Round((double)totalMsContradiction / contradictions, 2)
                : -1;

            //Console.WriteLine($"{rate}% contradiction rate ({contradictions}/{ATTEMPTS}) | avg {avgMs}ms | success {avgMsSuccess}ms | contradiction {avgMsContradiction}ms");

            csv.WriteLine($"{name},{imageFilename},{N},{size},{periodic},{symmetry},{heuristic},{ATTEMPTS},{contradictions},{successes},{rate},{avgMs},{avgMsSuccess},{avgMsContradiction}");

            results.Add((label, contradictions, successes));

            totalRuntimeMs += totalMs;
        }

        Console.WriteLine("\n--- summery - sorted by contradiction rate ---");
        results.Sort((a, b) => b.contradictions.CompareTo(a.contradictions));
        foreach (var (label, contradictions, successes) in results)
        {
            double rate = Math.Round((double)contradictions / ATTEMPTS * 100, 1);
            string difficulty = rate >= 55 ? "HIGH" : rate >= 15 ? "MEDIUM" : "LOW";
            Console.WriteLine($"[{difficulty,-6}] {rate}% - {label}");
        }

        Console.WriteLine($"\nTotal runtime was {totalRuntimeMs}");
        Console.WriteLine($"\nCSV saved to {Path.GetFullPath(csvPath)}");
    }
}