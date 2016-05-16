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

		biases[index-1] = algebra.CreateVectorWithMeanAndStdDeviation(value, 0, 1)//CreateVectorWithMeanAndStdDeviation CreateNormalizedVector(value)
	}

	network := &Network{weights: weights, biases: biases}//, costFunction:new(CrossEntrophyCostFunction), regularization: &L2Regularization{lambda:5}
	network.SetCostFunction(CrossEntrophy)
	network.SetRegularization(L2RegularizationType, 5.0)
	return network

}


