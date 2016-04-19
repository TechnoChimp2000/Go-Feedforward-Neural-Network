package network

import (
	"testing"
	"fmt"
	"strconv"
//	"time"
)

func TestHyperbolicTangensActivation(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetActivationFunction(HyperbolicTangens)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)

	if result[0] != -1.0625792E-01 || result[1] != 1.2098651E+00 {
		fmt.Println("network test with hyperbolic tangens activation function result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))
		fmt.Println("Expected: -1.0625792E-01 1.2098651E+00")
		t.Fail()
	}
}

func TestLogisticActivation(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetActivationFunction(Logistic)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)

	if result[0] != 4.708384E-01 || result[1] != 6.5444094E-01 {
		fmt.Println("network test with logistic activation function result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))
		fmt.Println("Expected: 4.708384E-01 6.5444094E-01")
		t.Fail()
	}
}

func TestLogisticActivationFunction_Derivative(t *testing.T) {

	a := new(LogisticActivationFunction)

	tests := []float32{-1,-0.99,-0.5,0,0.5,0.999,1}
	results := []float32{-2, -1.9701, -0.75, 0, 0.25, 0.0009989871,0 }

	for index, value := range tests {

		if results[index] != a.Derivative(value) {
			fmt.Printf("Expected derivative of %v is %v, but we got: %v\n", value, results[index], a.Derivative(value))
			t.Fail()

		}
	}
}

func TestHyperbolicTangentActivationFunction_Derivative(t *testing.T) {
	a := new(HyperbolicTangentActivationFunction)

	tests := []float32{-1, -0.5, 0, 0.5, 10}
	results := []float32{0.79195386, 1.02954, 1.1439333,  1.02954, 0.025172127}

	for index, value := range tests {

		if results[index] != a.Derivative(value) {
			fmt.Printf("Expected derivative of %v is %v, but we got: %v\n", value, results[index], a.Derivative(value))
			t.Fail()
		}
	}
}
