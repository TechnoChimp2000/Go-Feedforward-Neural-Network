package network2

import (
	"Go-Feedforward-Neural-Network/algebra"
	"math"
	"math/rand"
	"fmt"
)

type TrainingSet struct {
	trainingSamples []*TrainingSample
}

func(trainingSet *TrainingSet)shuffle(){
	for i := range trainingSet.trainingSamples {
		j := rand.Intn(i + 1)
		trainingSet.trainingSamples[i], trainingSet.trainingSamples[j] = trainingSet.trainingSamples[j], trainingSet.trainingSamples[i]
	}
}

func(trainingSet *TrainingSet)GetRandomMiniBatches(miniBatchSize int)[][]*TrainingSample{
	//fmt.Println("Minibatch size: ", miniBatchSize)

	trainingSet.shuffle()
	length := len(trainingSet.trainingSamples) / miniBatchSize
	result := make([][]*TrainingSample, length)
	z:=0
	for i := 0; i < length; i++ {
		result[i] = make([]*TrainingSample, miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			//fmt.Printf("Minibatch i: %d  j: %d \n",i,j )
			result[i][j] = trainingSet.trainingSamples[z]
			z++
		}

	}
	return result
}



type TrainingSample struct{
	Input  []float32
	Output []float32
}


type Network struct{
	weights []*algebra.Matrix
	biases [][]float32
}


func (network *Network)Feedforward(input []float32)[]float32{
	var result = input
	for i,_ := range network.weights{
		product := algebra.Multiply(network.weights[i], result)
		result = algebra.Vectorize(sigmoid, algebra.AddVectors(product, network.biases[i]))
	}
	return result
}

func sigmoid (input float32) float32 {
	return float32(1.0/(1.0+math.Exp(-float64(input))))
}


func (network *Network)Train(trainingSet *TrainingSet, epochs int, miniBatchSize int, eta float32, testSet *TrainingSet){
	for i:=0; i<epochs; i++{
		miniBatches := trainingSet.GetRandomMiniBatches(miniBatchSize)
		for j:=0; j<len(miniBatches);j++{
			network.updateMiniBatch(miniBatches[j], eta)
		}
		if testSet !=nil && len(testSet.trainingSamples)>0 {
			fmt.Printf("Epoch %d: %d / %d\n", i, network.evaluate(testSet), len(testSet.trainingSamples))

		}else{
			fmt.Printf("Epoch %d complete\n", i)
		}
	}
}




func(network *Network)updateMiniBatch(miniBatch []*TrainingSample, eta float32){
	nablaBiases := algebra.CreateZerofiedDoubleArray(network.biases)
	nablaWeights := algebra.CreateZerofiedMatrices(network.weights)

	for _, trainingSample := range miniBatch{
		deltaNablaBiases, deltaNablaWeights := network.backPropagate(trainingSample.Input, trainingSample.Output)
		nablaBiases = algebra.AddVectorArrays(nablaBiases, deltaNablaBiases)
		nablaWeights = algebra.AddMatriceArrays(nablaWeights, deltaNablaWeights)
	}

	factor := eta/(float32)(len(miniBatch))

	/**
	 * update biases
	 */
	network.biases = algebra.SubstractVectorArrays(network.biases, algebra.MultiplyVectorArrayWithNumber(nablaBiases, factor))

	/**
	 * update weights
	 */
	network.weights = algebra.SubstractMatriceArrays(network.weights, algebra.MultiplyMatrixArrayWithNumber(nablaWeights, factor))
}


func(network *Network)backPropagate(input, output []float32) ([][]float32, []*algebra.Matrix){

	nablaBiases := algebra.CreateZerofiedDoubleArray(network.biases)
	nablaWeights := algebra.CreateZerofiedMatrices(network.weights)

	activations := make([][]float32, len(network.biases)+1)
	activations[0] = input

	weightedInputs := algebra.CreateZerofiedDoubleArray(network.biases)

	activation := input

	for i,_ := range(network.biases){
		weightedInput := algebra.AddVectors(algebra.Multiply(network.weights[i], activation), network.biases[i])
		weightedInputs[i] = weightedInput

		activation = algebra.Vectorize(sigmoid, weightedInput)
		activations[i+1] = activation

	}

	delta := algebra.Hadamard(costDerivative(activations[len(activations)-1], output), sigmoidDerivative(weightedInputs[len(weightedInputs)-1]))

	nablaBiases[len(nablaBiases)-1] = delta

	nablaWeights[len(nablaWeights)-1] = algebra.MultiplyVectorWithTranspose(delta, activations[len(activations)-2])

	for i := 2; i<(len(network.biases)+1); i++ {


		weightedInput := weightedInputs[len(weightedInputs)-i]
		sigDerivative := sigmoidDerivative(weightedInput)

		delta = algebra.Hadamard(algebra.Multiply(algebra.TransposeMatrix(network.weights[len(network.weights)-i+1]), delta), sigDerivative)

		nablaBiases[len(nablaBiases)-i] = delta
		nablaWeights[len(nablaWeights)-i] = algebra.MultiplyVectorWithTranspose(delta, activations[len(activations)-i-1])

	}

	return nablaBiases, nablaWeights
}


func sigmoidDerivative(x []float32)[]float32{



	ones :=algebra.CreateArrayWithDefaultValue(len(x),1)
	sig := algebra.Vectorize(sigmoid, x)
	substraction := algebra.SubstractVectors(ones, sig)
	return algebra.Hadamard(sig, substraction)

}


func costDerivative(activations, output []float32)[]float32{
	return algebra.SubstractVectors(activations, output)
}


func(network *Network)evaluate(testSet *TrainingSet)int{
	successes :=0
	for i:=0; i<len(testSet.trainingSamples); i++{
		result := network.Feedforward(testSet.trainingSamples[i].Input)
		index := algebra.GetIndexOfMaxValue(result)
		if(testSet.trainingSamples[i].Output[index]>0){
			successes++
		}
	}

	return successes
}
