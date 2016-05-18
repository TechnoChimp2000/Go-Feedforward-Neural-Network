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

	input := []float64{3.14, 6, 7, 8, 8, 9}
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
	//nn := CreateNetwork([]int{784, 30, 10})
	network := CreateNetwork([]int{784, 30, 10})
	network.SetRegularization(SkipRegularizationType, 0)


	//nn.Train(GetTrainingSet(train_data), 30, 10, 0.1, GetTrainingSet(test_data))
	network.Train(getCustomTrainingSet(train_data, 5), 50, 5, 0.1, getCustomTrainingSet(train_data, 5))
	fmt.Printf("Finished training.\n")

}



func getCustomTrainingSet(training_data []network.TrainingSample, size int)*TrainingSet{
	trainingSamples := make ([]*TrainingSample, size)
	for i:=0;i<size;i++{
		example := (float64)(training_data[i].Input[0])
		fmt.Printf("ex: ", example)
		trainingSamples[i] = &TrainingSample{Input: convertArray(training_data[i].Input), Output: convertArray(training_data[i].Output)}
	}
	return &TrainingSet{trainingSamples: trainingSamples}
}
