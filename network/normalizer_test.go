package network

import (
	"testing"
	"fmt"
	"strconv"
)

func TestZscoreNormalizer(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetNormalizer(Zscore)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with z-score normalizer result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestSkipNormalizer(t *testing.T){

	network, trainingSamples := createSimpleNN()
	network.SetNormalizer(None)

	network.Train(trainingSamples)
	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test without normalizer result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}
