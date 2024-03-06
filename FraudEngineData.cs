using Microsoft.ML.Data;

namespace fraudengine
{
    public class FraudEngineDataAnomalyDetection
    {
        public class TransactionHistoryData
        {
            [LoadColumn(4)]
            public float Amount;

            [LoadColumn(5)]
            public string TimeStamp;
        }

        public class FraudEnginePrediction
        {
            //vector to hold alert,score,p-value values
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }
    }
}
