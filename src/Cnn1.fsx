open System

// ReLU activation function
let relu x = if x > 0.0 then x else 0.0

// 2D Convolution operation (without padding, stride 1)
let convolve2D (input: float[,]) (kernel: float[,]) =
    let inH = input.GetLength 0
    let inW = input.GetLength 1
    let kH = kernel.GetLength 0
    let kW = kernel.GetLength 1
    let outH = inH - kH + 1
    let outW = inW - kW + 1

    let output = Array2D.zeroCreate<float> outH outW

    for i in 0 .. outH - 1 do
        for j in 0 .. outW - 1 do
            let mutable sum = 0.0
            for ki in 0 .. kH - 1 do
                for kj in 0 .. kW - 1 do
                    sum <- sum + input.[i+ki, j+kj] * kernel.[ki,kj]
            output.[i,j] <- relu sum // activación ReLU
    output

// ====================
// Example: Edge Detection
// ====================

// 5x5 input image (simple pattern)
let image = array2D [
    [|0.0; 0.0; 1.0; 0.0; 0.0|]
    [|0.0; 1.0; 1.0; 1.0; 0.0|]
    [|1.0; 1.0; 1.0; 1.0; 1.0|]
    [|0.0; 0.0; 1.0; 0.0; 0.0|]
    [|0.0; 0.0; 1.0; 0.0; 0.0|]
]

// 3x3 Sobel-like kernel for edge detection
let kernel = array2D [
    [| -1.0; 0.0; 1.0 |]
    [| -1.0; 0.0; 1.0 |]
    [| -1.0; 0.0; 1.0 |]
]

let result = convolve2D image kernel

// Print result
printfn "Convolution Result:"
for i in 0 .. result.GetLength 0 - 1 do
    for j in 0 .. result.GetLength 1 - 1 do
        printf "%4.1f " result.[i,j]
    printfn ""
