package network

/*

// 1 - Load test data into trainData with loader
// 2 - Load the network from a file or train it yourself into n
// 3 - GetAccuaracy:

accuracy := n.GetAccuracy(trainData)
fmt.Printf("Accuracy of the trained network is: %v\n", &accuracy)

// Accuracy of the trained network is: 0.9827

*/



// helper functions related to statistics etc....
// RUNS THROUGH THE TRAINING SAMPLES AND RETURNS ACCURACY ( 1- accuaracy = error rate )
func (n *NeuralNetwork) GetAccuracy(trainData *[]TrainingSample) (accuracy float32) {

	var correctPredictions float32

	var total float32 = float32( len(*trainData) )
	for _, samples := range *trainData {

		predictionDetailed	:= n.Calculate(samples.Input)
		predictionNormalized 	:= Predict(predictionDetailed)

		isPredictionCorrect := CompareTwoSlicesFloat32(predictionNormalized, samples.Output)

		if isPredictionCorrect == true {
			correctPredictions++
		}

	}

	accuracy = correctPredictions / total

	return accuracy
}

// CHANGES (0.13 , 0.96, 0.02) INTO (0,1,0)
func Predict(input []float32) []float32 {

	var element float32
	output := input
	// find highest element
	for _,v := range input {

		if v > element {
			element = v
		}
	}

	// modify output
	for k,v := range output {
		if v == element {
			output[k] = 1
		} else {
			output[k] = 0
		}
	}
	return output
}

// COMPARES TWO VECTORS AND CHECKS IF THEY ARE THE SAME
func CompareTwoSlicesFloat32(input1, input2 []float32) bool {

	for k,v := range input1 {

		if input2[k] != v {
			return false
		}
	}
	return true
}

//[]network.TrainingSample



// The following three items are relevant for binary classification, so there is no need to write them down for now


// RUNS THROUGH THE TRAINING SAMPLES AND RETURNS PRECISION (true positives as a ratio of true positives + false negatives
func (n *NeuralNetwork) GetPrecision(trainData *[]TrainingSample) {}

// RUNS THROUGH THE TRAINING SAMPLES AND RETURNS RECALL
func (n *NeuralNetwork) GetRecall() {}

// Calculates F1 Score

func GetF1Score(precision, recall float32) {}