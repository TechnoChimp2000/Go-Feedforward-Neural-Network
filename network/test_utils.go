package network

import (

	"time"
"math/rand"
	"fmt"
)

func createRandomNN(inputOutputDim int, numOfLayers int, trainingSetDim int)(n *NeuralNetwork, trainingSet []TrainingSample){
	layers := make([]int,numOfLayers)
	for i := 0;i<numOfLayers;i++{
		layers[i] = inputOutputDim
	}
	fmt.Println("Layers are: ", layers)
	network := CreateNetwork(layers)
	randomTrainingSet := createRandomTrainingSet(inputOutputDim, trainingSetDim)
	return network, randomTrainingSet

}

func createRandomTrainingSet(inputOutputDim, trainingSetDim int) []TrainingSample{
	var trainingSet []TrainingSample

	for i:=0;i<trainingSetDim;i++{
		var trainingSample TrainingSample
		for j:=0;j<inputOutputDim;j++{
			trainingSample.Input = append(trainingSample.Input, getRandomZeroOrOne())
			trainingSample.Output = append(trainingSample.Output, getRandomZeroOrOne())
		}
		fmt.Println("Appending training sample: ", trainingSample)
		trainingSet = append(trainingSet, trainingSample)
	}


	return trainingSet

}

func getRandomZeroOrOne()float32{
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	return float32(r1.Intn(2))
}


func createSimpleNN() (n *NeuralNetwork, trainingSet []TrainingSample) {
	network := CreateNetwork([]int{2, 2, 2})

	trainingSamples := createSimpleTrainingSet()

	return network, trainingSamples

}

func createSimpleTrainingSet() []TrainingSample{
	trainingInput := []float32{0.05, 0.10}
	trainingOutput := []float32{0.01, 0.99}


	trainingSample := TrainingSample{Input: trainingInput, Output: trainingOutput}

	trainingSamples := []TrainingSample{trainingSample}
	return trainingSamples
}

func createXORTrainingSet() []TrainingSample{
	trainingSample1 := TrainingSample{Input: []float32{0.0, 0.0}, Output: []float32{0.0}}
	trainingSample2 := TrainingSample{Input: []float32{1.0, 1.0}, Output: []float32{0.0}}
	trainingSample3 := TrainingSample{Input: []float32{1.0, 0.0}, Output: []float32{1.0}}
	trainingSample4 := TrainingSample{Input: []float32{0.0, 1.0}, Output: []float32{1.0}}

	trainingSamples := []TrainingSample{trainingSample1, trainingSample2, trainingSample3, trainingSample4}

	return trainingSamples;
}