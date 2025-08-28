// Convolución simple 2D
let convolve (input: float[,]) (kernel: float[,]) =
    let inRows, inCols = input.GetLength(0), input.GetLength(1)
    let kRows, kCols = kernel.GetLength(0), kernel.GetLength(1)
    let outRows, outCols = inRows - kRows + 1, inCols - kCols + 1
    let output = Array2D.zeroCreate outRows outCols
    for i in 0 .. outRows - 1 do
        for j in 0 .. outCols - 1 do
            let mutable sum = 0.0
            for ki in 0 .. kRows - 1 do
                for kj in 0 .. kCols - 1 do
                    sum <- sum + input.[i+ki, j+kj] * kernel.[ki, kj]
            output.[i, j] <- sum
    output

// MaxPooling 2x2
let maxPool (input: float[,]) (poolSize: int) =
    let inRows, inCols = input.GetLength(0), input.GetLength(1)
    let outRows, outCols = inRows / poolSize, inCols / poolSize
    let output = Array2D.zeroCreate outRows outCols
    for i in 0 .. outRows - 1 do
        for j in 0 .. outCols - 1 do
            let mutable maxVal = System.Double.MinValue
            for pi in 0 .. poolSize - 1 do
                for pj in 0 .. poolSize - 1 do
                    maxVal <- max maxVal input.[i*poolSize+pi, j*poolSize+pj]
            output.[i, j] <- maxVal
    output

// Flatten: convierte matriz a vector
let flatten (input: float[,]) =
    input |> Seq.cast<float> |> Seq.toArray

// Capa densa (fully connected)
let dense (input: float[]) (weights: float[,]) (bias: float[]) =
    let outSize = weights.GetLength(0)
    let output = Array.zeroCreate outSize
    for i in 0 .. outSize - 1 do
        let mutable sum = 0.0
        for j in 0 .. input.Length - 1 do
            sum <- sum + input.[j] * weights.[i, j]
        output.[i] <- sum + bias.[i]
    output

// ------------------ DEMO ------------------
let input = array2D [ [1.0; 0.0; 1.0]
                      [0.0; 1.0; 0.0]
                      [1.0; 0.0; 1.0] ]

let kernel = array2D [ [1.0; 0.0]
                       [0.0; 1.0] ]

// Paso 1: convolución
let convOut = convolve input kernel
printfn "Convolution output:\n%A" convOut

// Paso 2: pooling
let pooled = maxPool convOut 2
printfn "After pooling:\n%A" pooled

// Paso 3: flatten
let flat = flatten pooled
printfn "Flatten:\n%A" flat

// Paso 4: capa densa (ejemplo con 1 neurona de salida)
let weights = array2D [ [0.5] ]   // 1x1 porque la entrada es un solo valor
let bias = [| 0.1 |]

let output = dense flat weights bias
printfn "Dense layer output:\n%A" output
