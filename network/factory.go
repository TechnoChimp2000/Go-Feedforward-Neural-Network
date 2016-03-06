package network

import (
	"math/rand"
	"time"
)


func CreateNeuronLayer(neuronNumber int, bias float32) NeuronLayer{

	var neurons []*Neuron
	for i :=0; i< neuronNumber; i++ {
		neurons = append(neurons, new(Neuron))
	}
	return NeuronLayer{Neurons: neurons, Bias: bias}
}

func CreateNetwork(topology []int, biasUnits []float32, epsilon float32 , trainingSamples []TrainingSample, learningRate float32, precision float32) (n *NeuralNetwork) {

	if epsilon == 0 {
		panic("Epsilon is 0, please assign to a non-zero value.")
	}

	// EXAMPLE CreateNetwork ( float32{2,3,3,4}, []int{ 0.45, 0.3, 0.5, 0 }

	var neuronLayers []*NeuronLayer
	// BUILD LAYERS and append them

	for i, numberOfNeurons := range topology {

		layer := CreateNeuronLayer( numberOfNeurons , biasUnits[i] )
		neuronLayers = append(neuronLayers, & layer)

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

	// initialize weights, randomly between 0 and 1 and multiply it by epsilon
	// since it's the same loop, we might as well define the []InputConnections there as well
	initializeWeightsAndInputConnections(neuronLayers, epsilon)

	// declare neural network
	neuronNetwork := NeuralNetwork{NeuronLayers: neuronLayers,
		ActivationFunction: new(LogisticActivationFunction),
		TrainingSet: trainingSamples,
		LearningRate: precision,
		Precision: precision,
		trainer:new(OnlineTrainer)}



	return &neuronNetwork
}

func initializeWeightsAndInputConnections(neuronLayers []*NeuronLayer, epsilon float32) {

	// use a different seed every time for weight initialization
	// TODO: maybe we can use a static seed during testing for result comparison?
	source := rand.NewSource(time.Now().UnixNano())
	random := rand.New(source)

	for i, layer := range neuronLayers {
		if i==0 {
			continue
		}
		for _, neuronInThisLayer := range layer.Neurons {
			for _, neuronInPreviousLayer := range neuronLayers[i-1].Neurons {
				weight := random.Float32()
				w := &Connection{From: neuronInPreviousLayer, To: neuronInThisLayer, Weight: weight*epsilon}

				neuronInThisLayer.InputConnections = append( neuronInThisLayer.InputConnections, w )
			}
		}
	}
}
