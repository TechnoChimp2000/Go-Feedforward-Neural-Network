package network

import (
	"math/rand"
	"time"
)


func createNeuronLayer(neuronNumber int, bias float32) NeuronLayer{

	var neurons []*Neuron
	for i :=0; i< neuronNumber; i++ {
		neurons = append(neurons, new(Neuron))
	}
	return NeuronLayer{Neurons: neurons, Bias: bias}
}

func CreateNetwork(topology []int) (n *NeuralNetwork) {


	// EXAMPLE CreateNetwork ( float32{2,3,3,4}, []int{ 0.45, 0.3, 0.5, 0 }

	var neuronLayers []*NeuronLayer
	// BUILD LAYERS and append them

	biasUnits := createRandomBiases(len(topology))

	for i, numberOfNeurons := range topology {

		layer := createNeuronLayer( numberOfNeurons , biasUnits[i] )
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
	initializeWeightsAndInputConnections(neuronLayers)

	// declare neural network
	neuronNetwork := NeuralNetwork{neuronLayers: neuronLayers,
		ActivationFunction: new(LogisticActivationFunction),
		learningRate: 0.02,
		precision: 0.005,
		trainer:new(OnlineTrainer)}



	return &neuronNetwork
}

func createRandomBiases(length int)[]float32{
	source := rand.NewSource(time.Now().UnixNano())
	random := rand.New(source)

	result := make([]float32, length)
	for i := 0; i<length; i++{
		if(i != length-1){
			result[i] = random.Float32()
		}
	}
	return result
}

func initializeWeightsAndInputConnections(neuronLayers []*NeuronLayer) {

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
				w := &Connection{From: neuronInPreviousLayer, To: neuronInThisLayer, Weight: weight}

				neuronInThisLayer.InputConnections = append( neuronInThisLayer.InputConnections, w )
			}
		}
	}
}
