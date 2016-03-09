package network


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