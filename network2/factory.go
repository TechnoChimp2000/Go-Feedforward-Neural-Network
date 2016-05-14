package network2

import "Go-Feedforward-Neural-Network/algebra"


func CreateNetwork(topology []int)*Network{
	weights := make ([]*algebra.Matrix, len(topology)-1)

	for index, value := range topology{
		if index == 0{
			continue
		}
		weightMatrix := algebra.CreateNormalizedMatrix(value, topology[index - 1])
		weights[index-1] = weightMatrix
	}

	biases := make([][]float32, len(topology)-1)
	for index, value := range topology {
		if index == 0 {
			continue
		}

		biases[index-1] = algebra.CreateNormalizedVector(value)
	}

	network := &Network{weights: weights, biases: biases}
	return network

}


