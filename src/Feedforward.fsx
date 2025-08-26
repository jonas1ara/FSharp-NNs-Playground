open System

// Define the structure of the network

type Layer =
    {
        Weights: float[,]
        Biases: float[]
    }

type Network =
    {
        Hidden: Layer
        Output: Layer
    }

// Auxiliary functions

// Sigmoide and its derivative
let sigmoid x = 1.0 / (1.0 + exp(-x))
let sigmoidDerivative x = x * (1.0 - x) // Note: here x is already sigmoid(x)

// Random initialization
let rand = Random()

let initLayer inputSize outputSize =
    {
        Weights = Array2D.init outputSize inputSize (fun _ _ -> rand.NextDouble() * 2.0 - 1.0)
        Biases = Array.init outputSize (fun _ -> rand.NextDouble() * 2.0 - 1.0)
    }

let initNetwork inputSize hiddenSize outputSize =
    {
        Hidden = initLayer inputSize hiddenSize
        Output = initLayer hiddenSize outputSize
    }

// Forward pass

let dot (weights: float[,]) (inputs: float[]) (biases: float[]) =
    Array.init (weights.GetLength 0) (fun i ->
        let mutable sum = biases.[i]
        for j in 0 .. weights.GetLength 1 - 1 do
            sum <- sum + weights.[i,j] * inputs.[j]
        sum
    )

let forward network input =
    // Hidden layer
    let hiddenRaw = dot network.Hidden.Weights input network.Hidden.Biases
    let hiddenActivated = hiddenRaw |> Array.map sigmoid

    // Output layer
    let outputRaw = dot network.Output.Weights hiddenActivated network.Output.Biases
    let outputActivated = outputRaw |> Array.map sigmoid

    hiddenActivated, outputActivated

// Backpropagation

let train (network: Network) (inputs: float[][]) (targets: float[][]) learningRate epochs =
    let mutable net = network

    for epoch in 1 .. epochs do
        for i in 0 .. inputs.Length - 1 do
            let x = inputs.[i]
            let y = targets.[i]

            // Forward pass
            let hidden, output = forward net x

            // Output layer errors and deltas
            let outputErrors = Array.map2 (fun t o -> t - o) y output
            let outputDeltas = Array.map2 (fun err o -> err * sigmoidDerivative o) outputErrors output

            // Hidden layer errors and deltas
            let hiddenErrors =
                Array.init hidden.Length (fun j ->
                    seq { for k in 0 .. outputDeltas.Length-1 -> outputDeltas.[k] * net.Output.Weights.[k,j] }
                    |> Seq.sum
                )

            let hiddenDeltas = Array.map2 (fun err h -> err * sigmoidDerivative h) hiddenErrors hidden

            // Update output weights
            for k in 0 .. net.Output.Weights.GetLength 0 - 1 do
                for j in 0 .. net.Output.Weights.GetLength 1 - 1 do
                    net.Output.Weights.[k,j] <- net.Output.Weights.[k,j] + learningRate * outputDeltas.[k] * hidden.[j]
                net.Output.Biases.[k] <- net.Output.Biases.[k] + learningRate * outputDeltas.[k]

            // Update hidden weights
            for j in 0 .. net.Hidden.Weights.GetLength 0 - 1 do
                for k in 0 .. net.Hidden.Weights.GetLength 1 - 1 do
                    net.Hidden.Weights.[j,k] <- net.Hidden.Weights.[j,k] + learningRate * hiddenDeltas.[j] * x.[k]
                net.Hidden.Biases.[j] <- net.Hidden.Biases.[j] + learningRate * hiddenDeltas.[j]

    net

// ====================
// Example: XOR
// ====================

let trainingInputs =
    [|
        [| 0.0; 0.0 |]
        [| 0.0; 1.0 |]
        [| 1.0; 0.0 |]
        [| 1.0; 1.0 |]
    |]

let trainingTargets =
    [|
        [| 0.0 |]
        [| 1.0 |]
        [| 1.0 |]
        [| 0.0 |]
    |]

let inputSize = 2
let hiddenSize = 2   // XOR needs at least 2 hidden neurons
let outputSize = 1

let network = initNetwork inputSize hiddenSize outputSize

let trained = train network trainingInputs trainingTargets 0.5 5000

// Test
for inp, target in Array.zip trainingInputs trainingTargets do
    let _, output = forward trained inp
    printfn "Input: %A -> Pred: %.4f (target %.1f)" inp output.[0] target.[0]
