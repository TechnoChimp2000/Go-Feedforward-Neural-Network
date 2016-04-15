package network

import (
	random "math/rand"
	//"time"
	"math"
	//"fmt"
)


func createNeuronLayer(neuronNumber int) NeuronLayer{

	var neurons []*Neuron
	for i :=0; i< neuronNumber; i++ {
		neurons = append(neurons, new(Neuron))
	}

	return NeuronLayer{Neurons: neurons}
}

func CreateNetwork(topology []int) (n *NeuralNetwork) {

	// EXAMPLE CreateNetwork ([]int{ 2, 2, 3}

	var neuronLayers []*NeuronLayer
	// BUILD LAYERS and append them

	for _, numberOfNeurons := range topology {

		layer := createNeuronLayer( numberOfNeurons )
		neuronLayers = append(neuronLayers, & layer)

	}

	// BUILD THE CONNECTIONS BETWEEN LAYERS
	for layerIndex, layer := range neuronLayers[:len(neuronLayers)-1] {
		for _, neuronInThisLayer := range layer.Neurons {
			for _, neuronInNextLayer := range neuronLayers[layerIndex+1].Neurons {
				neuronInThisLayer.ConnectedToInNextLayer = append( neuronInThisLayer.ConnectedToInNextLayer, neuronInNextLayer)
			}
		}
	}

	// initialize weights and InputConnections
	initializeWeightsAndInputConnections(neuronLayers)
	// initialize a bias and delta of each neuron
	initializeNeuronBiasAndDelta(neuronLayers)

	// declare neural network
	neuronNetwork := NeuralNetwork{neuronLayers: neuronLayers}

	// set some default values for the created neural network
	neuronNetwork.SetPrecision(Medium)
	neuronNetwork.SetLearningRate(Normal, 0 )
	neuronNetwork.SetTrainerMode(Offline)
	neuronNetwork.SetActivationFunction(Logistic)
	neuronNetwork.SetNormalizer(Zscore)
	neuronNetwork.SetCostFunction(CrossEntrophy)

	return &neuronNetwork
}

func initializeWeightsAndInputConnections(neuronLayers []*NeuronLayer) {

	// use a different seed every time for weight initialization
	// TODO: maybe we can use a static seed during testing for result comparison?
	//this is not ok, random is the same for all weights
	/*source := rand.NewSource(time.Now().UnixNano())
	random := rand.New(source)*/

	for i, layer := range neuronLayers {
		if i==0 {
			continue
		}
		for _, neuronInThisLayer := range layer.Neurons {

			for _, neuronInPreviousLayer := range neuronLayers[i-1].Neurons {
				//weight := random.Float32()
				weight := calculateWeight(neuronLayers[i-1])
				w := &Connection{From: neuronInPreviousLayer, To: neuronInThisLayer, Weight: weight}

				neuronInThisLayer.InputConnections = append( neuronInThisLayer.InputConnections, w )

			}
		}
	}
}

func initializeNeuronBiasAndDelta(neuronLayers []*NeuronLayer) {

	for _, layer := range neuronLayers[1:] {
		for _, neuron := range layer.Neurons {

			neuron.Bias	= 2*random.Float32()-1 // i want to generate the bias between -1 and 1.
			neuron.Delta	= 0
		}
	}
}

/**
 * weight is from interval (−1,√d)(1,√d), where d is the number of inputs to a given neuron
 * see: http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
 */
func calculateWeight(layerFrom *NeuronLayer)float32{
	numOfInputs := len(layerFrom.Neurons)
	interval := 1 / float32(math.Sqrt(float64(numOfInputs)))
	return calculateRandomNumInInterval(interval)
}

func calculateRandomNumInInterval(interval float32)float32{
	return (random.Float32()*interval*2)-interval
}
