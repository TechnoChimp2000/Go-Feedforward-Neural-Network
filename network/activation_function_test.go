package network

import (
	"testing"
	"fmt"
	"strconv"
)

func TestHyperbolicTangensActivation(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetActivationFunction(HyperbolicTangens)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with hyperbolic tangens activation function result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestLogisticActivation(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetActivationFunction(Logistic)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with logistic activation function result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}
