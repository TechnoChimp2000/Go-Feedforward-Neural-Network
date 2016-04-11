package network

/*
this part of the package can store a trained network to a file, or load it from a file and use it for predictions

1) Create a network fully from file

	load_from_this_file := "/files/file.json"

	//network.SetLoadFileLocation(load_from_this_file)
	n := network.LoadNetworkFromFile(load_from_this_file)

2) Store trained network to a file (n --> trained NeuralNetwork )

	save_to_this_file := "/files/file.json"

	n.SaveToFile(save_to_this_file)

*/

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

)


type storedNetwork struct {
	Topology []int
	UnrolledWeights []float32
	ActivationFunction string
	Biases []float32
}


func (n *NeuralNetwork) SaveToFile( fp string ) () {

	// check for argument

	if len(fp) == 0  {
		fmt.Printf("no argument")
	}

	// create a slice for weights
	var weights []float32

	// unroll the weights into the slice for storing purposes
	for _, layer := range n.neuronLayers[1:] {
		for _, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {
				weights = append(weights, inputConnection.Weight)
			}
		}
	}

	// get topology information from the neural network
	var topology []int
	for _, layer := range n.neuronLayers {
		topology = append(topology, len(layer.Neurons))
	}

	// get activation function
	activationFunction := n.GetActivationFunction()

	// get biases
	var biases []float32
	for _, layer := range n.neuronLayers {
		biases = append(biases, layer.Bias)
	}


	// Create a network to be stored

	sNetwork := storedNetwork{
		Topology: 		topology,
		UnrolledWeights: 	weights,
		ActivationFunction: 	activationFunction,
		Biases: 		biases,
	}

	// encode the data structure to json
	b, err := json.Marshal(sNetwork)
	if err != nil {
		fmt.Println("Error: ", err)
	}

	// store to a file
	ioutil.WriteFile( fp, b, 0644 )
}

func LoadNetworkFromFile( fp string ) (n *NeuralNetwork) {

	//read the file
	file, err := ioutil.ReadFile(fp)
	if err != nil {
		fmt.Printf("File Read error: %v\n", err)
		os.Exit(1)
	}

	var sNetwork storedNetwork
	json.Unmarshal(file , &sNetwork)

	// now we can build the network
	n = CreateNetwork( sNetwork.Topology )

	// set the ActivationFunction
	if (sNetwork.ActivationFunction == "HyperbolicTangensActivationFunction") {
		n.SetActivationFunction(HyperbolicTangens)
	} else {
		n.SetActivationFunction(Logistic)
	}



	// insert all the weights to the appropriate connection
	i := 0

	for _, layer := range n.neuronLayers[1:] {
		for _, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {
				inputConnection.Weight = sNetwork.UnrolledWeights[i]
				i++
			}
		}
	}

	// insert all the biases to the appropriate layer
	i = 0

	for _, layer := range n.neuronLayers {
		layer.Bias = sNetwork.Biases[i]
		i++
	}

	return n
}

