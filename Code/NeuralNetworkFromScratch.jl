using Parameters
using Plots
using LinearAlgebra
using ProgressMeter
using Statistics
using LaTeXStrings


"""
    main()

Initializes and trains neural network to approximate a function f(x). 

"""
function main()

    # Define function that we would like to learn with our neural network
    f(x) = x^2                       # Function
    xRange = (min = -1.0, max = 1.0) # Range of input values x for which we would like to learn f(x)

    # Generate a sample of trainingData
    sampleSize = 100
    rawInputs = [[(xRange.max - xRange.min) * rand() + xRange.min] for ii in 1:sampleSize] # Draw inputs from uniform distribtuion
    rawOutputs = [[f(rawInputs[ii]...)] for ii in 1:sampleSize] # Compute outputs for each input
    trainingData = prepareTrainingData(rawInputs, rawOutputs) # Normalize inputs and outputs

    # Initialize neural network with default settings and initialize gradient struct
    NN = NeuralNetwork()
    NNGradient = NeuralNetworkGradient(NN) # this will be used during gradient descent

    # Feed inputs in the untrained neural network
    output = feedforward(NN, [1.0])
    println("Output of untrained neural network for input = 1.0: ", output)

    # Train the neural network 
    epochs = 1000
    showProgress = true
    trainingLosses = zeros(epochs) # Initialize vectors to keep track of training
    p = Progress(epochs; desc = "Training...", color = :grey, barlen = 0) # Creates a progress bar
    
    @time for ii in 1:epochs

        trainNeuralNetwork!(NN, NNGradient, trainingData.inputs, trainingData.outputs)

        # Update progress indicator
        if showProgress
            trainingLosses[ii] = loss(NN, trainingData.inputs, trainingData.outputs)
            next!(p; showvalues = [(:loss, trainingLosses[ii]), (:logloss, log10.(trainingLosses[ii]))], valuecolor = :grey)
        end

    end

    # Feed inputs in the trained neural network
    output = feedforward(NN, [1.0])
    println("Output of trained neural network for input = 1.0: ", output)

    # Plot output for trained neural network
    p1 = plot(xRange.min:0.01:xRange.max, x -> f(x), label = "f(x)", linewidth = 2)
    plot!(xRange.min:0.01:xRange.max, x -> feedforward(NN, [x], 
        trainingData.inputNormFactors, trainingData.outputNormFactors)[1], label = "f_NN(x)", linestyle = :dash, linewidth = 2, color = :black)
    title!("Trained Neural Network")
    xlabel!("Input")
    ylabel!("Output")

    # Plot training loss
    p2 = plot(1:epochs, log10.(trainingLosses), label = "", linewidth = 2)
    title!("Training Loss")
    xlabel!("Epoch")
    ylabel!(L"\log_{10}(\textrm{Loss})")

    pp = plot(p1, p2, layout = 2, size = (800, 270))

    return pp

end


"""
    NeuralNetwork()

Basic settings of the neural network with single hidden layer.

"""
@with_kw mutable struct NeuralNetwork

    # Number of nodes of input, output and hidden layer
    nInputs::Int64 = 1
    nHidden::Int64 = 5
    nOutputs::Int64 = 1

    # Weights (Initialized by drawing from standard normal distribution)
    w1::Array{Float64,2} = randn(nHidden, nInputs)
    w2::Array{Float64,2} = randn(nOutputs, nHidden)

    # Biases (Initialized to zero)
    b1::Array{Float64,1} = zeros(nHidden)
    b2::Array{Float64,1} = zeros(nOutputs)

    # Regularization parameter
    λ::Float64 = 0.0

    # Learning rate during gradient descent
    learningSpeed::Float64 = 0.01

    # Activation function
    activationFunction::Symbol = :sigmoid # Supported: :sigmoid, :softplus, :relu

end


"""
    feedforward(NN::NeuralNetwork, inputs)

Computes the outputs of neural network NN for given inputs.

"""
function feedforward(NN::NeuralNetwork, inputs)

    # NN contains all required information regarding weights, biases and activation functions
    
    # Compute the output of the hidden neurons
    hidden = activation.(Ref(NN), NN.w1 * inputs .+ NN.b1)  # Note: the dot-notation means that 
                                                            # operations or functions are applied 
                                                            # elementwise (this is also called 
                                                            # broadcasting)

    # Compute the output of the neural network
    outputs = NN.w2 * hidden .+ NN.b2

    return outputs

end


"""
    feedforward(NN::NeuralNetwork, rawInputs, inputNormFactors, outputNormFactors)

Computes the outputs of neural network NN for given inputs. This also automatically
takes care of input and output normalization

"""
function feedforward(NN::NeuralNetwork, rawInputs, inputNormFactors, outputNormFactors)
    
    # Normalize inputs
    inputs = [normalize(inputNormFactors[jj], rawInputs[jj]) for jj in 1:length(rawInputs)]
    
    # Feedforward
    normalizedOutputs = feedforward(NN, inputs)
    
    # Denormalize outputs
    outputs = [denormalize(outputNormFactors[jj], normalizedOutputs[jj]) for jj in 1:length(normalizedOutputs)]
    
    return outputs

end


"""
    activation(NN::NeuralNetwork, x)

Activation function used by the Neural Network.

"""
function activation(NN::NeuralNetwork, x)

    if NN.activationFunction == :softplus
        return log(1+exp(x))
    elseif NN.activationFunction == :relu
        return (x < 0.0) ? 0.0 : x
    elseif NN.activationFunction == :sigmoid
        return 1/(1+exp(-x))
    end

end


"""
    activationPrime(NN, x)

Derivative of activation function used by the Neural Network.

"""
function activationPrime(NN::NeuralNetwork, x)

    if NN.activationFunction == :softplus
        return 1/(1+exp(-x))
    elseif NN.activationFunction == :relu
        return (x < 0.0) ? 0.0 : 1
    elseif NN.activationFunction == :sigmoid
        return (1/(1+exp(-x))) * (1 - 1/(1+exp(-x)))
    end

end


"""
    trainNeuralNetwork!(NN::NeuralNetwork, NNGradient, inputs, outputs)

Updates the weights of the neural network.

"""
function trainNeuralNetwork!(NN::NeuralNetwork, NNGradient, inputs, outputs)

    for ii in 1:length(inputs)

        # Compute the gradient
        computeGradient!(NN, NNGradient, inputs[ii], outputs[ii]) # More efficient but harder to understand
        # computeGradientSimplified!(NN, NNGradient, inputs[ii], outputs[ii]) # Slower but easier to understand

        # Determine learning rate (more advanced techniques possible)
        learn = NN.learningSpeed

        # Update the weights of the neural network
        @. NN.w1 = NN.w1 - learn * NNGradient.w1
        @. NN.w2 = NN.w2 - learn * NNGradient.w2
        @. NN.b1 = NN.b1 - learn * NNGradient.b1
        @. NN.b2 = NN.b2 - learn * NNGradient.b2

    end

    nothing

end


"""
    NeuralNetworkGradient()

Struct that holds the gradient of the loss function of the neural network.

"""
struct NeuralNetworkGradient
    w1::Array{Float64,2}
    w2::Array{Float64,2}
    b1::Array{Float64,1}
    b2::Array{Float64,1}
end


"""
    NeuralNetworkGradient(NN::NeuralNetwork)

Initializes NeuralNetworkGradient struct based on dimensions of neural network NN.

"""
function NeuralNetworkGradient(NN::NeuralNetwork)

    NeuralNetworkGradient(
        zeros(size(NN.w1)),
        zeros(size(NN.w2)),
        zeros(size(NN.b1)),
        zeros(size(NN.b2))
    )

end


"""
    computeGradient!(NN::NeuralNetwork, NNGradient, inputs, outputs)

Computes gradient of loss function. Used when training the neural network.

"""
function computeGradient!(NN::NeuralNetwork, NNGradient, inputs, outputs)

    # Get the sample size
    sampleSize = size(inputs, 1)

    # Reset the gradients
    NNGradient.w1 .= zeros(size(NN.w1))
    NNGradient.w2 .= zeros(size(NN.w2))
    NNGradient.b1 .= zeros(size(NN.b1))
    NNGradient.b2 .= zeros(size(NN.b2))

    # Preallocate matrices to keep total allocations low
    z_Lm1 = zeros(NN.nHidden)
    a_Lm1 = zeros(NN.nHidden)
    z_L = zeros(NN.nOutputs)
    a_L = zeros(NN.nOutputs)
    δ_L = zeros(NN.nOutputs)
    δ_Lm1 = zeros(NN.nHidden)

    for ii in 1:sampleSize

        # Feed forward
        z_Lm1 .= mul!(z_Lm1, NN.w1, inputs[ii]) .+ NN.b1
        a_Lm1 .= activation.(Ref(NN), z_Lm1)
        z_L .= mul!(z_L, NN.w2, a_Lm1) .+ NN.b2
        a_L .= z_L

        # Backpropagation
        @. δ_L = (a_L - outputs[ii])
        δ_Lm1 .= mul!(δ_Lm1, NN.w2', δ_L) .* activationPrime.(Ref(NN), z_Lm1)
 
        # Compute gradient and add it to the sum
        mul!(NNGradient.w2,  δ_L, a_Lm1', 1.0, 1.0)
        mul!(NNGradient.w1,  δ_Lm1, inputs[ii]', 1.0, 1.0)
        NNGradient.b2 .+= δ_L
        NNGradient.b1 .+= δ_Lm1

    end

    # Divide by sample size
    rdiv!(NNGradient.w2, sampleSize)
    rdiv!(NNGradient.w1, sampleSize)
    rdiv!(NNGradient.b2, sampleSize)
    rdiv!(NNGradient.b1, sampleSize)

    # Add regularization term
    @. NNGradient.w1 = 2 * (NNGradient.w1 + NN.λ / sampleSize * NN.w1)
    @. NNGradient.w2 = 2 * (NNGradient.w2 + NN.λ / sampleSize * NN.w2)
    @. NNGradient.b1 = 2 * NNGradient.b1
    @. NNGradient.b2 = 2 * NNGradient.b2

    nothing

end


"""
    computeGradientSimplified!(NN::NeuralNetwork, NNGradient, inputs, outputs)

Computes gradient of loss function. Used when training the neural network. This
is a simplified version of computeGradient!, which only works for stochastic
gradient descent and does not take advantage of in-place matrix multiplactions.

"""
function computeGradientSimplified!(NN::NeuralNetwork, NNGradient, inputs, outputs)
    
    # NNGradient is a struct of matrices holding the gradients for weights and biases	

    # Feed forward
    z_Lm1 = NN.w1 * inputs .+ NN.b1         # Summation hidden layer
    a_Lm1 = activation.(Ref(NN), z_Lm1)     # Activation hidden layer
    z_L = NN.w2 * a_Lm1 .+ NN.b2            # Summation output layer
    a_L = z_L                               # Activation output layer (linear)

    # Backpropagation
    δ_L = (a_L .- outputs)  
    δ_Lm1 = (NN.w2' * δ_L) .* activationPrime.(Ref(NN), z_Lm1)
 
    # Compute gradient
    NNGradient.w2 .= δ_L * a_Lm1'
    NNGradient.w1 .= δ_Lm1 * inputs'
    NNGradient.b2 .= δ_L
    NNGradient.b1 .= δ_Lm1

    # Add regularization term
    @. NNGradient.w1 = 2 * (NNGradient.w1 + NN.λ * NN.w1)
    @. NNGradient.w2 = 2 * (NNGradient.w2 + NN.λ * NN.w2)
    @. NNGradient.b1 = 2 * NNGradient.b1
    @. NNGradient.b2 = 2 * NNGradient.b2
    # Note @. applies a dot to each operation and function call on a line 
    # (i.e. makes everything into elementwise operations) 

    nothing

end


"""
    loss(NN, inputs, outputs)

Computes loss of neural network for given dataset.

"""
function loss(NN::NeuralNetwork, inputs, outputs)
    return loss(NN, NN.w1, NN.w2, NN.b1, NN.b2, inputs, outputs)
end


"""
    loss(NN, w1, w2, b1, b2, inputs, outputs)

Computes loss of neural network for given dataset using alternative weights and biases.

"""
function loss(NN::NeuralNetwork, w1, w2, b1, b2, inputs, outputs)

    # Get the sample size
    sampleSize = size(inputs, 1)

    # Initialiize the loss
    L = 0.0

    # Preallocate matrices to keep total allocations low
    z_Lm1 = zeros(NN.nHidden)
    a_Lm1 = zeros(NN.nHidden)
    z_L = zeros(NN.nOutputs)
    a_L = zeros(NN.nOutputs)
    tmp = zeros(NN.nOutputs)

    for ii in 1:length(inputs)

        # Feed forward
        z_Lm1 .= mul!(z_Lm1, w1, inputs[ii]) .+ b1
        a_Lm1 .= activation.(Ref(NN), z_Lm1)
        z_L .= mul!(z_L, w2, a_Lm1) .+ b2
        a_L .= z_L

        # Evalute loss for input
        @. tmp = (outputs[ii] - a_L)^2
        L += tmp[1] / sampleSize # Assumes that there is only one output

    end

    # Add regularization term
    L = L + NN.λ / sampleSize * (sum(w1.^2) + sum(w2.^2))

    return L

end


"""
    prepareTrainingData(inputs, outputs; IONormalization = :standard)

Normalizes inputs and outputs for training of the neural network.

"""
function prepareTrainingData(inputs, outputs; IONormalization = :standard)

    inputNormFactors = []
    outputNormFactors = []
    normalizedInputs = similar(inputs)
    normalizedOutputs = similar(outputs)

    # Get normalization factors for inputs
    for ii in 1:length(inputs[1])

        # Collect ii input
        sim = [x[ii] for x in inputs]

        # Compute normalization
        normFactors = getNormalization(IONormalization, sim)

        # Save normalization
        push!(inputNormFactors, normFactors)

    end

    # Get normalization factors for inputs
    for ii in 1:length(outputs[1])

        # Collect ii output
        sim = [x[ii] for x in outputs]

        # Compute normalization
        normFactors = getNormalization(IONormalization, sim)

        # Save normalization
        push!(outputNormFactors, normFactors)

    end

    # Normalize inputs and outputs
    for ii in 1:length(inputs)
        normalizedInputs[ii] = [normalize(inputNormFactors[jj], inputs[ii][jj]) for jj in 1:length(inputs[ii])]
        normalizedOutputs[ii] = [normalize(outputNormFactors[jj], outputs[ii][jj]) for jj in 1:length(outputs[ii])]
    end

    # Collect all information for the training data
    trainingData = (inputNormFactors = inputNormFactors, 
                    outputNormFactors = outputNormFactors, 
                    inputs = normalizedInputs, 
                    outputs = normalizedOutputs)

    return trainingData

end


"""
    getNormalization(P, sim)

Returns tuple with location and scale factor that are used for normalization.

"""
function getNormalization(IONormalization, sim)

    if IONormalization == :minMax01

        normFactors = (location = minimum(sim), scale = maximum(sim)-minimum(sim))

    elseif IONormalization == :minMax11

        normFactors = (location = (minimum(sim)+maximum(sim))/2 , scale = (maximum(sim)-minimum(sim))/2)

    elseif IONormalization == :minMax44

        normFactors = (location = (minimum(sim)+maximum(sim))/2 , scale = (1/4) * (maximum(sim)-minimum(sim))/2)
    
    elseif IONormalization == :standard

        normFactors = (location = mean(sim), scale = std(sim))
    
    elseif IONormalization == :none

        normFactors = (location = 0.0, scale = 1.0)
    
    end

    return normFactors

end


"""
    normalize(normFactors, x)

Normalizes x using location and scale parameter in normFactors.

"""
function normalize(normFactors, x)

    return (x - normFactors.location) / normFactors.scale 

end


"""
    denormalize(normFactors, x)

Denormalizes x using location and scale parameter in normFactors.

"""
function denormalize(normFactors, x)

    return x * normFactors.scale + normFactors.location

end
