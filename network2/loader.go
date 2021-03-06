package network2

import "Go-Feedforward-Neural-Network/network"

func GetTrainingSet(training_data []network.TrainingSample)*TrainingSet{
	trainingSamples := make ([]*TrainingSample, len(training_data))
	for i,training_sample :=range(training_data){
		trainingSamples[i] = &TrainingSample{Input: convertArray(training_sample.Input), Output: convertArray(training_sample.Output)}
	}
	return &TrainingSet{trainingSamples: trainingSamples}
}
