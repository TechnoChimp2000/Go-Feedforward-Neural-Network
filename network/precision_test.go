package network

import (
	"testing"
	"fmt"
	"strconv"
)

func TestRoughPrecision(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetPrecision(Rough)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with rough precision result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestMediumPrecision(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetPrecision(Medium)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with medium precision result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestHighPrecision(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetPrecision(High)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with high precision result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestVeryHighPrecision(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetPrecision(VeryHigh)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with very high precision result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}
