/*
TODO: final goal --> take a set of parameters, output trained model.

The goal is to have a basic gradient descent implemented for standard parameters.
What are the standard parameters?

1) cost function we will minimize
2) gradients or partial derivatives of our cost function with respect to parameters
3) propagation function --> calculates z foor one neuron
4) activation function --> calculates g(z) for one neuron
5) forward propagation method --> loop through all the training samples and that will give us the error terms
6) back-propagation    method --> loop through all the training samples to collect all the derivatives
7) update of all weights simultaneously

*/
package gd

import (
	"math"
)
/*
Just some copy pasting for easier thinking

type NeuralNetwork struct{
	neuronLayers []NeuronLayer
	learningRate float32
	trainingSet []TrainingSample
	connectionLayers []ConnectionLayer
}*/

// some data struct as a starting point. Will change latter to something smarter once I understand better what is it that I want to do.
type GradientDecentParameters struct {
	NNparameters	map[string]int // here we can dynamically collect information from main.
	LearningRate	float32

}

///////////
// Point 1 - we outline the etype of the CostFunction, but don't define it
type CostFunctionType func( []float64 ) float64

// Point 2 - we implement Gradient desent based on CostFunction, and Neural Network parammeters. Both can be fetched from the main
func GradientDesent(fn CostFunctionType, X[]float64  ) []float64 {

	var weights_optimal []float64

	// here is where we will have the core of the product.
	// we need to dynamically construct all the elemments of optimization,
	// TODO: How would you do it with objective programing?
	// TODO: What parts of the network package could be taken out, and fed into the gradient desent here?
	// TODO: I would like to use the network package to dynamically construct the cost function, and gradients, which should then find their way back here
	//
	//

	fn(X) // forward propagation

	// backward propagation
	// updating
	// repeat until ....


	return weights_optimal
}
// Point 3 -- this is just an example of a function that is of type CostFunctionType. We do not need to have this function
// in this package. In fact, it is better to pass it to GradientDesent from main(), and it can get there from network package perhaps? .
// I would like Gradient desent implemented in a fully independent way that can then be used for any optimization given initial parameters

func PropagationFunction ( X[]float64 ) float64 {
	var output float64
	for i:= range X {
		output +=X[i]
	}
	return output

}

///////////

// Generic sigmoid or logit function
// TODO: This should also probably go out of this package, and be only defined as a function type here.
//
func Sigmoid( w float64) float64 {
	z:= 1 + ( 1 + math.Exp(-float64(w)) )
	return float64(z)
}