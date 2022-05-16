using System;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;


// See https://aka.ms/new-console-template for more information

namespace ML_project1
{
    class FeedbackTrainingData
    {
        //input data structure (of algorithm)

        //label (classify based on this (output of the algo)) 
        [Column("Label", Order = 0)]
        public bool Label { get; set; }

        //feature (input to the algo)
        [Column(Order = 1)]
        public string FeedbackText { get; set; } = string.Empty;

    }

    class FeedbackPrediction
    {
        // output data structure (of algorithm)
        [Column("PredictedLabel")]
        public bool PredictedLabel { get; set; }
    }

    class Program
    {   
        static List<FeedbackTrainingData> trainingData = 
            new List<FeedbackTrainingData>();

        static List<FeedbackTrainingData> testData =
            new List<FeedbackTrainingData>();

        static void LoadTrainingData()
        {
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText =  "This is good",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "This is horrible",
                Label = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad and hell",
                Label = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "this nice but can be better",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad bad",
                Label = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "till now it looks nice",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "its very Average",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad horrible",
                Label = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "quiet average",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "sooo nice",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "average and ok",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "well ok ok",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "very good",
                Label = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "quiet average",
                Label = true
            });
        }

        static void LoadTestData()
        {
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "good",
                Label= true
            });
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "nice",
                Label = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "horrible terrible",
                Label = false
            });
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "nice",
                Label = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "shitty",
                Label = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "average",
                Label = true
            });
        }
        static void Main(string[] args)
        {
            // step 1 : Load the training data
            LoadTrainingData();

            //step 2 : Create object of MLContext
            var mlContext = new MLContext();

            // step 3 : convert training data to IDataView
            IDataView dataView = mlContext.Data.LoadFromEnumerable(trainingData);
            
            // step 4 : We need to create the pipeline 
            // define workflows in it.
            // Transform feedbacktext which is a feature to a vector
            // Define the trainer
            var options = new FastTreeBinaryTrainer.Options
            {
                NumberOfTrees = 50,
                NumberOfLeaves = 50,
                MinimumExampleCountPerLeaf = 1
            };
            var pipeline = 
                mlContext.Transforms.Text.FeaturizeText("Features","FeedbackText")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(options));

            // step 5 : Train the model and we want the model out
            var model = pipeline.Fit(dataView);

            // step 6 : Load the test data and run the test data 
            // to check our model's accuracy
            LoadTestData();

            // step 7 : convert test data to IDataView
            IDataView testDataView = mlContext.Data.LoadFromEnumerable(testData);
            //get predictions by feeding testData to model
            var predictions = model.Transform(testDataView);
            // evaluate the predictions against label (isGood) to check model's accuracy
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);

            // step 8 : use the model
            Console.WriteLine("Enter a feedback string");
            string feedbackString = Console.ReadLine().ToString();
            // create prediction function from the model
            var predictionFunction = 
                mlContext.Model.CreatePredictionEngine
                <FeedbackTrainingData, FeedbackPrediction>(model);

            var feedbackInput = new FeedbackTrainingData()
            {
                FeedbackText = feedbackString,
            };
            var feedbackPredicted = predictionFunction.Predict(feedbackInput);
            Console.WriteLine("Predicted :-" + feedbackPredicted.PredictedLabel);
            Console.ReadLine();
        }

    }
}
