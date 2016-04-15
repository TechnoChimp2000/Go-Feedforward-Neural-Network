package network

import (
	"testing"
	"fmt"
	"strconv"
)

func TestFastLearningRate(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetLearningRate(Fast,0)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with fast learing rate result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestNormalLearningRate(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetLearningRate(Normal,0)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with normal learing rate result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestSlowLearningRate(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetLearningRate(Slow,0)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with slow learing rate result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestVerySlowLearningRate(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetLearningRate(VerySlow,0)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with very slow learing rate result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}