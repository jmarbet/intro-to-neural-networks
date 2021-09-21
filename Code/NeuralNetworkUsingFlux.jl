using Flux
using Plots
using LinearAlgebra
using ProgressMeter
using Statistics
using LaTeXStrings


function mainFlux()

    # Define function that we would like to learn with our neural network
    f(x) = x^2                       # Function
    xRange = (min = -1.0, max = 1.0) # Range of input values x for which we would like to learn f(x)
    
    # Generate a sample of trainingData
    sampleSize = 100
    rawInputs = [[(xRange.max - xRange.min) * rand() + xRange.min] for ii in 1:sampleSize] # Draw inputs from uniform distribtuion
    rawOutputs = [[f(rawInputs[ii]...)] for ii in 1:sampleSize] # Compute outputs for each input
    trainingData = zip(rawInputs, rawOutputs) # Initialize training dataset for stochastic gradient descent (creates an iterator of tuples with inputs and outputs)
    #trainingData = Flux.Data.DataLoader((hcat(rawInputs...), hcat(rawOutputs...)), batchsize=1) # Alternatively you can use the dataloader, 
                                                                                                 # which allows you to specifify the batchsize for
                                                                                                 # minibatch (or batch) gradient descent
    # Note: we are not normalizing the inputs/outputs in this example
    
    # Note: To inspect the batches in trainingData you can loop over them as follows
    # for (x, y) in trainingData
    #     display(x) # Displays all inputs that are used to compute gradient for one step in the gradient descent
    #     # display(y) # This displays the corresponding outputs
    # end

    # Define the neural network layers (this defines a function called model(x))
    model = Chain(
        Dense(1, 5, σ), # Input-Hidden (sigmoid activation)
        Dense(5, 1)     # Hidden-Output (linear activation)
        )

    # Note the definition above is shorthand for the following
    # W1 = rand(5, 1)
    # b1 = rand(5)
    # layer1(x) = W1 * x .+ b1
    #
    # W2 = rand(1, 5)
    # b2 = rand(1)
    # layer2(x) = W2 * x .+ b2
    # 
    # model(x) = layer2(σ.(layer1(x)))

    # We can feedforward by simply calling model(x)
    println("Output of untrained neural network for input = 1.0: ", model([1.0]))

    # Define loss function and weights 
    loss(x, y) = Flux.Losses.mse(model(x), y)
    ps = Flux.params(model)

    # Train the neural network 
    epochs = 1000
    showProgress = true
    opt = Descent(0.01) # learning rate
    trainingLosses = zeros(epochs) # Initialize vectors to keep track of training
    p = Progress(epochs; desc = "Training...", color = :grey, barlen = 0) # Creates a progress bar
    
    @time for ii in 1:epochs
        
        Flux.train!(loss, ps, trainingData, opt)
      
        # Update progress indicator
        if showProgress
            trainingLosses[ii] = mean([loss(x,y) for (x,y) in trainingData])
            next!(p; showvalues = [(:loss, trainingLosses[ii]), (:logloss, log10.(trainingLosses[ii]))], valuecolor = :grey)
        end

    end

    # Feed inputs in the trained neural network
    println("Output of trained neural network for input = 1.0: ", model([1.0]))

    # Plot output for trained neural network
    p1 = plot(xRange.min:0.01:xRange.max, x -> f(x), label = "f(x)", linewidth = 2)
    plot!(xRange.min:0.01:xRange.max, x -> model([x])[1], label = "f_NN(x)", linestyle = :dash, linewidth = 2, color = :black)
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

