package main

import (
	"fmt"
	"Go-Feedforward-Neural-Network/loader"

	"Go-Feedforward-Neural-Network/network2"
)

func main(){

	fmt.Printf("Started loading training data..\n")

	pimages := &loader.Create{Filepath : "xdata/mnist/train-images.idx3-ubyte" }
	plabels := &loader.Create{Filepath : "xdata/mnist/train-labels.idx1-ubyte" }

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
	ptestimages := &loader.Create{Filepath : "xdata/mnist/t10k-images.idx3-ubyte"}
	ptestlabels := &loader.Create{Filepath : "xdata/mnist/t10k-labels.idx1-ubyte"}

	ptestimages.Load()
	ptestlabels.Load()

	test_data := loader.CreateTrainingSet(ptestimages, ptestlabels, 10)
	fmt.Printf("Num of testing items: %v\n", len(test_data))

	fmt.Printf("Training...\n")

	nn := network2.CreateNetwork([]int{784, 30, 10})


	nn.Train(network2.GetTrainingSet(train_data), 30, 10, 0.01, network2.GetTrainingSet(test_data))

	fmt.Printf("Finished training.\n")

}
