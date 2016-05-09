package network2

import (
"Go-Feedforward-Neural-Network/algebra"
"math"
)

type Network struct{
	weights []*algebra.Matrix
	biases [][]float32
}

/*"""Return the output of the network if ``a`` is input."""
for b, w in zip(self.biases, self.weights):
a = sigmoid(np.dot(w, a)+b)
return a*/

func (network *Network)Feedforward(input []float32)[]float32{
	var result = input
	for i,_ := range network.weights{
		product := algebra.Multiply(network.weights[i], result)
		result = algebra.Vectorize(sigmoid, algebra.AddVectors(product, network.biases[i]))
	}
	return result
}

func sigmoid (input float32) float32 {
	return float32(1.0/(1.0+math.Exp(-float64(input))))
}
