package network2

import (
	"testing"
	"fmt"
	"Go-Feedforward-Neural-Network/loader"
	"Go-Feedforward-Neural-Network/network"
)


func TestFeedforward(t *testing.T){
	topology := []int{4,3,5, 6, 7}
	network := CreateNetwork(topology)

	input := []float32{3.14, 6, 7, 8, 8, 9}
	result := network.Feedforward(input)
	fmt.Printf("Result: ", result)
}

func TestDigitsRecognition(t *testing.T){
	fmt.Printf("Started loading training data..\n")

	pimages := &loader.Create{Filepath : "../xdata/mnist/train-images.idx3-ubyte" }
	plabels := &loader.Create{Filepath : "../xdata/mnist/train-labels.idx1-ubyte" }

	pimages.Load()
	plabels.Load()

	fmt.Printf("Some image:\n")
	fmt.Printf("[")
	rand_index:=1810;
	for _,value := range pimages.Data[rand_index]{
		fmt.Printf("%v,", value)
	}
	fmt.Printf("]\n")

	fmt.Printf("It's label:\n%v\n", plabels.Data[rand_index])

	train_data := loader.CreateTrainingSet(pimages, plabels, 10)
	fmt.Printf("Num of training items: %v\n", len(train_data))

	fmt.Printf("Started loading testing data..\n")
	ptestimages := &loader.Create{Filepath : "../xdata/mnist/t10k-images.idx3-ubyte"}
	ptestlabels := &loader.Create{Filepath : "../xdata/mnist/t10k-labels.idx1-ubyte"}

	ptestimages.Load()
	ptestlabels.Load()

	test_data := loader.CreateTrainingSet(ptestimages, ptestlabels, 10)
	fmt.Printf("Num of testing items: %v\n", len(test_data))

	fmt.Printf("Training...\n")
	nn := CreateNetwork([]int{784, 30, 10})

	//trainingSet *TrainingSet, epochs int, miniBatchSize int, eta float32, testSet *TrainingSet
	nn.Train(getTrainingSet(train_data), 30, 10, 3.0, getTrainingSet(test_data))
	fmt.Printf("Finished training.\n")

}

func getTrainingSet(training_data []network.TrainingSample)*TrainingSet{
	trainingSamples := make ([]*TrainingSample, len(training_data))
	for i,training_sample :=range(training_data){
		trainingSamples[i] = &TrainingSample{Input: training_sample.Input, Output: training_sample.Output}
	}
	return &TrainingSet{trainingSamples: trainingSamples}
}
