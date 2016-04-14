package network

import "math"

type CostFunction interface{
	calculateTotalError(actual []float32, output []float32) float32
	calculateWeightDeltaInLastLayer(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32)
	calculateWeightDelta(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32)
}

type QuadraticCostFunction struct{}

func (q *QuadraticCostFunction) calculateTotalError(actual []float32, output []float32) float32 {
	var retval float32 = 0.0
	for outputIndex, singleOutput := range output{
		factor := singleOutput - actual[outputIndex]
		retval += 0.5 * factor * factor
	}
	return retval/float32(len(output))
}

func (q *QuadraticCostFunction) calculateWeightDeltaInLastLayer(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	// First calculate the last layer deltas
	//n.neuronLayers[i].deltas[neuronIndex] =  neuron.output - trainingSampleOutput[neuronIndex]

	// Second, in same neuron for loop, calculate F1 and F2
	factor1 := neuron.output - trainingSampleOutput[neuronIndex]
	factor2 := n.activationFunction.Derivative(neuron.output)

	//n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] = factor1 * factor2

	n.neuronLayers[neuronLayerIndex].Neurons[neuronIndex].NewDelta = factor1 * factor2
}

func (q *QuadraticCostFunction) calculateWeightDelta(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	var factor1 float32


	for _, neuronInNextLayer := range neuron.ConnectedToInNextLayer {

		var weight float32 	= neuronInNextLayer.InputConnections[neuronIndex].Weight
		//factor1 += n.neuronLayers[neuronLayerIndex].deltas[neuronInNextLayerIndex] * weight
		factor1 += neuronInNextLayer.NewDelta * weight
	}

	factor2	:= n.activationFunction.Derivative(neuron.output)

	//n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] =  factor1 * factor2
	n.neuronLayers[neuronLayerIndex].Neurons[neuronIndex].NewDelta = factor1 * factor2
}


/*
func (q *QuadraticCostFunction) calculateWeightDeltaInLastLayerNEW(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	// First calculate the last layer deltas
	//n.neuronLayers[i].deltas[neuronIndex] =  neuron.output - trainingSampleOutput[neuronIndex]

	// Second, in same neuron for loop, calculate F1 and F2
	factor1 := neuron.output - trainingSampleOutput[neuronIndex]
	factor2 := n.activationFunction.Derivative(neuron.output)

	//n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] = factor1 * factor2

	n.neuronLayers[neuronLayerIndex].Neurons[neuronIndex].NewDelta = factor1 * factor2
}

func (q *QuadraticCostFunction) calculateWeightDeltaNEW(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	var factor1 float32


	for _, neuronInNextLayer := range neuron.ConnectedToInNextLayer {

		var weight float32 	= neuronInNextLayer.InputConnections[neuronIndex].Weight
		//factor1 += n.neuronLayers[neuronLayerIndex].deltas[neuronInNextLayerIndex] * weight
		factor1 += neuronInNextLayer.NewDelta * weight
	}

	factor2	:= n.activationFunction.Derivative(neuron.output)

	//n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] =  factor1 * factor2
	n.neuronLayers[neuronLayerIndex].Neurons[neuronIndex].NewDelta = factor1 * factor2
}

*/


type CrossEntrophyCostFunction struct{}

func (c *CrossEntrophyCostFunction) calculateTotalError(actual []float32, output []float32) float32 {
	var retval float32 = 0.0
	for outputIndex, singleOutput := range output{
		factor := (singleOutput * float32(math.Log(float64(actual[outputIndex])))) + ((1 - singleOutput) * float32(math.Log(float64(1-actual[outputIndex]))))
		retval += factor
	}
	retval = -retval
	return retval/float32(len(output))
}

func (c *CrossEntrophyCostFunction) calculateWeightDeltaInLastLayer(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	// First calculate the last layer deltas
	//n.neuronLayers[i].deltas[neuronIndex] =  neuron.output - trainingSampleOutput[neuronIndex]

	// Second, in same neuron for loop, calculate F1 and F2

	factor1 := neuron.output - trainingSampleOutput[neuronIndex]
	//factor2 := n.activationFunction.Derivative(neuron.output)

	n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] = factor1
}

func (c *CrossEntrophyCostFunction) calculateWeightDelta(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	var factor1 float32


	for neuronInNextLayerIndex, neuronInNextLayer := range neuron.ConnectedToInNextLayer {

		var weight float32 	= neuronInNextLayer.InputConnections[neuronIndex].Weight
		factor1 += n.neuronLayers[neuronLayerIndex].deltas[neuronInNextLayerIndex] * weight
	}

	//factor2 	:= n.activationFunction.Derivative(neuron.output)

	n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] =  factor1
}

/*
func (q *CrossEntrophyCostFunction) calculateWeightDeltaNEW(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	var factor1 float32


	for neuronInNextLayerIndex, neuronInNextLayer := range neuron.ConnectedToInNextLayer {

		var weight float32 	= neuronInNextLayer.InputConnections[neuronIndex].Weight
		factor1 += n.neuronLayers[neuronLayerIndex].deltas[neuronInNextLayerIndex] * weight
	}

	factor2 	:= n.activationFunction.Derivative(neuron.output)

	n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] =  factor1 * factor2
}

func (q *CrossEntrophyCostFunction) calculateWeightDeltaInLastLayerNEW(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32) {
	// First calculate the last layer deltas
	//n.neuronLayers[i].deltas[neuronIndex] =  neuron.output - trainingSampleOutput[neuronIndex]

	// Second, in same neuron for loop, calculate F1 and F2

	factor1 := neuron.output - trainingSampleOutput[neuronIndex]
	factor2 := n.activationFunction.Derivative(neuron.output)

	n.neuronLayers[neuronLayerIndex-1].deltas[neuronIndex] = factor1 * factor2
}

*/

