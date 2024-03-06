using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using static fraudengine.FraudEngineDataAnomalyDetection;

namespace fraudengine.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class AnomaliesController : ControllerBase
    {
        private readonly MLContext _mlContext;
        private static readonly string _dataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "AllTransactionHistory.csv");
        private const int _docSize = 36;

        public AnomaliesController(MLContext mlContext)
        {
            _mlContext = mlContext; 
        }

        [HttpGet("DetectFraud")]
        public IActionResult DetectSpike(float amount)
        {
            var dataView = _mlContext.Data.LoadFromTextFile<TransactionHistoryData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            var predictions = DetectSpike(_mlContext, _docSize, dataView);

            // Filter predictions where the first index is 1, indicating a spike
            var spikes = predictions.Where(p => p.Prediction[0] == 1).ToArray();

            var ranges = SelectMaxMinMid(spikes);
            var response = EvaluateTransactionAmount(ranges, amount);
            return Ok(response);
        }
        /*
         
        [HttpGet("DetectChangepoint")]
        public IActionResult DetectChangepoint()
        {
            var dataView = _mlContext.Data.LoadFromTextFile<TransactionHistoryData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            var predictions = DetectChangepoint(_mlContext, _docSize, dataView);
            return Ok(predictions);
        }
         */

        private IEnumerable<FraudEnginePrediction> DetectSpike(MLContext mlContext, int docSize, IDataView allTransactionHistory)
        {
            Console.WriteLine("Detect temporary changes in pattern");

            // STEP 2: Set the training algorithm
            // <SnippetAddSpikeTrainer>
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(FraudEnginePrediction.Prediction), inputColumnName: nameof(TransactionHistoryData.Amount), confidence: 95, pvalueHistoryLength: docSize / 4);
            // </SnippetAddSpikeTrainer>

            // STEP 3: Create the transform
            // Create the spike detection transform
            Console.WriteLine("=============== Training the model ===============");
            // <SnippetTrainModel1>
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView());
            // </SnippetTrainModel1>

            Console.WriteLine("=============== End of training process ===============");
            //Apply data transformation to create predictions.
            // <SnippetTransformData1>
            IDataView transformedData = iidSpikeTransform.Transform(allTransactionHistory);
            // </SnippetTransformData1>

            // <SnippetCreateEnumerable1>
            var predictions = mlContext.Data.CreateEnumerable<FraudEnginePrediction>(transformedData, reuseRowObject: false);
            // </SnippetCreateEnumerable1>

            // <SnippetDisplayHeader1>
            Console.WriteLine("Alert\tScore\tP-Value");
            // </SnippetDisplayHeader1>

            // <SnippetDisplayResults1>
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");
            // </SnippetDisplayResults1>
            return predictions;
        }
        /*
         
        private IEnumerable<FraudEnginePrediction> DetectChangepoint(MLContext mlContext, int docSize, IDataView allTransactionHistory)
        {
            // Changepoint detection code adapted from the console app
            // Similar to the static method in the original code, but returns the predictions instead of printing them.
            // Ensure you adapt the original console app's DetectChangepoint method to work here.
        }
         */

        private IDataView CreateEmptyDataView()
        {
            IEnumerable<TransactionHistoryData> enumerableData = new List<TransactionHistoryData>();
            return _mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        private (FraudEnginePrediction max, FraudEnginePrediction min, FraudEnginePrediction mid) SelectMaxMinMid(IEnumerable<FraudEnginePrediction> spikes)
        {
            // Ensure there are spikes to process
            if (!spikes.Any())
            {
                throw new InvalidOperationException("No spikes provided.");
            }

            // Sort the spikes based on the p.Prediction[1] value
            var sortedSpikes = spikes.OrderBy(p => p.Prediction[1]).ToList();

            // Get the max, min, and mid values
            var max = sortedSpikes.Last(); // Last element after sorting will have the maximum value
            var min = sortedSpikes.First(); // First element after sorting will have the minimum value
            FraudEnginePrediction mid;

            // Handling for the middle value
            if (sortedSpikes.Count == 1)
            {
                // If there's only one element, it is also the mid
                mid = sortedSpikes[0];
            }
            else
            {
                // Calculate the index for the middle element
                var midIndex = sortedSpikes.Count / 2;
                mid = sortedSpikes[midIndex];
            }

            return (max, min, mid);
        }

        private object EvaluateTransactionAmount((FraudEnginePrediction max, FraudEnginePrediction min, FraudEnginePrediction mid) predictions, float amount)
        {
            var (maxPrediction, minPrediction, midPrediction) = predictions;

            // If amount is greater than the max prediction value
            if (amount > maxPrediction.Prediction[1])
            {
                return new
                {
                    status = 11,
                    reason = "Unusual amount discovered with high rating",
                    action = "report"
                };
            }
            // If amount is greater than the mid prediction value
            else if (amount > midPrediction.Prediction[1] && amount < maxPrediction.Prediction[1])
            {
                return new
                {
                    status = 02,
                    reason = "Unusual amount discovered with medium rating",
                    action = "monitor"
                };
            }
            // If amount is less than the min prediction value
            else if (amount < minPrediction.Prediction[1])
            {
                return new
                {
                    status = 01,
                    reason = "Unusual amount discovered with low rating",
                    action = "monitor"
                };
            }
            // If amount is between mid and min prediction values
            else if (amount >= minPrediction.Prediction[1] && amount <= midPrediction.Prediction[1])
            {
                return new 
                {
                    status = 00,
                    reason = "Safe transaction",
                    action = "process"
                };
            }

            // Default case if none of the above conditions match
            // This could be adjusted based on further requirements
            return new
            {
                status = -1,
                reason = "Evaluation not conclusive",
                action = "review"
            };
        }

    }
}
