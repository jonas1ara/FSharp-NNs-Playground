open System

// Define the perceptron type
type Perceptron =
    {
        weights: float array
        bias: float
    }

// Activation function (in this case, the heaviside function)
let activationFunction x =
    if x > 0.0 then 1.0 else 0.0

// Initializing a perceptron with random weights and bias
let initializePerceptron inputSize =
    let random = Random()
    {
        weights = Array.init inputSize (fun _ -> random.NextDouble())
        bias = random.NextDouble()
    }

// Calculate the output of the perceptron for a given input
let predict perceptron input =
    let weightedSum = Array.sum (Array.map2 (*) perceptron.weights input)
    activationFunction (weightedSum + perceptron.bias)

// Train the perceptron with a dataset
let train (perceptron: Perceptron) (inputs: float array array) (targets: float array) (learningRate: float) (epochs: int) =
    let mutable perceptron = perceptron
    for epoch in 1 .. epochs do
        for i in 0 .. inputs.Length - 1 do
            let input = inputs.[i]
            let target = targets.[i]
            let prediction = predict perceptron input
            let error = target - prediction

            // Update weights and bias
            let updatedWeights =
                Array.map2 (fun w x -> w + learningRate * error * x) perceptron.weights input
            let updatedBias = perceptron.bias + learningRate * error

            // Construct a new perceptron with updated weights and bias
            perceptron <- { perceptron with weights = updatedWeights; bias = updatedBias }

// Example of use
let inputSize = 2
let perceptron = initializePerceptron inputSize

let trainingInputs = [|
    [| 0.0; 0.0 |]
    [| 0.0; 1.0 |]
    [| 1.0; 0.0 |]
    [| 1.0; 1.0 |]
|]

let trainingTargets = [| 0.0; 0.0; 0.0; 1.0 |]

let learningRate = 0.1
let epochs = 1000

train perceptron trainingInputs trainingTargets learningRate epochs

// Trained perceptron test
let testInput = [| 1.0; 1.0 |]
let prediction = predict perceptron testInput

printfn "Prediction for inputs [%f, %f]: %f" testInput.[0] testInput.[1] prediction


