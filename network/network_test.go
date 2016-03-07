package network

import (
	"testing"
	"fmt"
	"strconv"
)

type CallbackReceiver struct{}
func (c CallbackReceiver) ReceiveInfo(info string){
	fmt.Println(info)

}



func TestSimpleNN(t *testing.T){



	network := CreateNetwork([]int{2, 2, 2})

	trainingSamples := createSimpleTrainingSet()

	network.Train(trainingSamples)


	result := network.Calculate(trainingSamples[0].Input)
	fmt.Println("network test with factory result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))

}

func TestXOROnline(t *testing.T){

	network := CreateNetwork([]int{2, 2, 1})

	xorTrainingSet := createXORTrainingSet()

	network.Train(xorTrainingSet)


	result1 := network.Calculate([]float32{0.0, 0.0})
	fmt.Println("network test XOR (0,0) result: "+strconv.FormatFloat(float64(result1[0]), 'E', -1, 32))

	result2 := network.Calculate([]float32{1.0, 1.0})
	fmt.Println("network test XOR (1,1) result: "+strconv.FormatFloat(float64(result2[0]), 'E', -1, 32))

	result3 := network.Calculate([]float32{0.0, 1.0})
	fmt.Println("network test XOR (0,1) result: "+strconv.FormatFloat(float64(result3[0]), 'E', -1, 32))

	result4 := network.Calculate([]float32{1.0, 0.0})
	fmt.Println("network test XOR (1,0) result: "+strconv.FormatFloat(float64(result4[0]), 'E', -1, 32))


}


func createXORTrainingSet() []TrainingSample{
	trainingSample1 := TrainingSample{Input: []float32{0.0, 0.0}, Output: []float32{0.0}}
	trainingSample2 := TrainingSample{Input: []float32{1.0, 1.0}, Output: []float32{0.0}}
	trainingSample3 := TrainingSample{Input: []float32{1.0, 0.0}, Output: []float32{1.0}}
	trainingSample4 := TrainingSample{Input: []float32{0.0, 1.0}, Output: []float32{1.0}}

	trainingSamples := []TrainingSample{trainingSample1, trainingSample2, trainingSample3, trainingSample4}

	return trainingSamples;
}

func createSimpleTrainingSet() []TrainingSample{
	trainingInput := []float32{0.05, 0.10}
	trainingOutput := []float32{0.01, 0.99}


	trainingSample := TrainingSample{Input: trainingInput, Output: trainingOutput}

	trainingSamples := []TrainingSample{trainingSample}
	return trainingSamples
}
