package network

import (
	"math/rand"
	"time"
)


func CreateNeuronLayer(neuronNumber int, NumberOfInputConnections int, bias float32) NeuronLayer{

	var neurons []*Neuron
	for i :=0; i< neuronNumber; i++ {
		neurons = append(neurons, new(Neuron))
	}
	return NeuronLayer{Neurons: neurons, Bias: bias, NumberOfInputConnections:NumberOfInputConnections}
}

func CreateNetwork(topology []int, biasUnits []float32, epsilon float32 ) (n *NeuralNetwork) {

	if epsilon == 0 {
		panic("Epsilon is 0, please assign to a non-zero value.")
	}

	// EXAMPLE CreateNetwork ( float32{2,3,3,4}, []int{ 0.45, 0.3, 0.5, 0 }

	var neuronLayers []*NeuronLayer
	// BUILD LAYERS and append them

	for i, numberOfNeurons := range topology {

		if i == 0 {
			layer := CreateNeuronLayer( numberOfNeurons, 0, biasUnits[i] )
			neuronLayers = append(neuronLayers, & layer)
		} else {
			layer := CreateNeuronLayer( numberOfNeurons , topology[i-1] * numberOfNeurons , biasUnits[i] )
			neuronLayers = append(neuronLayers, & layer)
		}
	}

	// BUILD THE CONNECTIONS BETWEEN LAYERS
	// TODO: test this forloop for different topologies
	for layerIndex, layer := range neuronLayers[:len(neuronLayers)-1] {
		for _, neuronInThisLayer := range layer.Neurons {
			for _, neuronInNextLayer := range neuronLayers[layerIndex+1].Neurons {
				neuronInThisLayer.ConnectedToInNextLayer = append( neuronInThisLayer.ConnectedToInNextLayer, neuronInNextLayer)
			}
		}
	}

	// declare neural network
	neuronNetwork := NeuralNetwork{NeuronLayers: neuronLayers, ActivationFunction: new(LogisticActivationFunction)}

	// initialize weights, randomly between 0 and 1 and multiply it by epsilon
	// since it's the same loop, we might as well define the []InputConnections there as well
	neuronNetwork.InitializeWeightsAndInputConnections(epsilon)

	return &neuronNetwork
}

func (n * NeuralNetwork) InitializeWeightsAndInputConnections( epsilon float32) {

	// use a different seed every time for weight initialization
	// TODO: maybe we can use a static seed during testing for result comparison?
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)

	for i, layer := range n.NeuronLayers[1:] {
		for _, neuronInThisLayer := range layer.Neurons {
			for _, neuronInPreviousLayer := range n.NeuronLayers[i].Neurons {
				weight := r1.Float32()
				w := &Connection{From: neuronInPreviousLayer, To: neuronInThisLayer, Weight: weight*epsilon}

				neuronInThisLayer.InputConnections = append( neuronInThisLayer.InputConnections, w )
			}
		}
	}
}
