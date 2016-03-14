package network

import (
	"strconv"
	"fmt"
)

type Trainer interface{
	train(n *NeuralNetwork, trainingSet        []TrainingSample)
}


type OnlineTrainer struct{}

func (o *OnlineTrainer) train(n *NeuralNetwork, trainingSet        []TrainingSample){
	if n.debug {
		fmt.Println("OnlineTrainer start")
	}

	//TODO validate input
	totalSamplesTrained := 0
	for {
		var exit  bool = false
		for  range trainingSet {

			currentlyLearningSample := trainingSet[totalSamplesTrained]
			actual := n.feedForward(currentlyLearningSample.Input)
			_error := n.costFunction.calculateTotalError(actual, currentlyLearningSample.Output)
			if n.debug {
				fmt.Println("Error: ",_error)
			}

			/**
			 * if error is small enough check previous training samples if they are still trained in neural network
			 * ..otherwise start from beginning
			 */
			if _error < n.precision {
				totalSamplesTrained++

				if n.debug {
					fmt.Println("Total samples trained are: " + strconv.Itoa(totalSamplesTrained))
				}

				/**
				 * test if previous training samples are still trained..
				 * ..otherwise start training from beginning
				 */
				if(n.testNeuralNetwork(totalSamplesTrained-1, trainingSet)) {

					/**
					 * if all training samples are learned finish learning
					 */
					if totalSamplesTrained == len(trainingSet) {
						if n.debug {
							fmt.Println("Done")
						}
						exit = true
						break;
					}
				}else{
					if n.debug {
						fmt.Println("Restarting")
					}
					totalSamplesTrained = 0;
				}

			}else{
				/**
				 * apply backpropagation if training sample isn't trained yet
				 */
				n.backPropagate(currentlyLearningSample.Output)

				/*if n.debug {
					fmt.Println("Deltas: ", deltas)
				}*/

				n.updateWeightsFromDeltas()
			}

		}

		if(exit){
			break
		}

	}

}

type OfflineTrainer struct{}

func (o *OfflineTrainer) train( n *NeuralNetwork, trainingSet        []TrainingSample){
	panic("Not implemented yet")
}

func (w *NeuralNetwork) propagate(inputConnections []*Connection) float32{
	var result float32 = 0.0
	for _,inputConnection := range inputConnections{
		result += inputConnection.From.output * inputConnection.Weight
		//fmt.Println(inputConnection.From.output, inputConnection.Weight)
	}

	return result
}


func (n* NeuralNetwork) testNeuralNetwork(numSamples int, trainingSet        []TrainingSample) bool{
	for _, trainingSample := range trainingSet[:numSamples] {


		actual := n.feedForward(trainingSample.Input)
		_error := n.costFunction.calculateTotalError(actual, trainingSample.Output)

		if _error >= n.precision {
			return false
		}
	}
	return true

}

func (n *NeuralNetwork) backPropagate(trainingSampleOutput []float32) {  // deltas[indexLayer][indexNeuron]



	for i := len(n.neuronLayers) - 1; i > 0; i-- {
		/**
		 * processing last layer of neuron connections
		 */
		if(i == len(n.neuronLayers) - 1){

			/*var wg sync.WaitGroup

			wg.Add(n.NeuronLayers[i].NumberOfInputConnections)*/

			for neuronIndex, neuron := range n.neuronLayers[i].Neurons {


				n.costFunction.calculateWeightDeltaInLastLayer(n, neuron, neuronIndex, i, trainingSampleOutput)
			}

		} else {
			for neuronIndex, neuron := range n.neuronLayers[i].Neurons {

				n.costFunction.calculateWeightDelta(n, neuron, neuronIndex, i, trainingSampleOutput)
			}
		}
	}

}

func (n *NeuralNetwork) updateWeightsFromDeltas() {

	for indexLayer, layer := range n.neuronLayers[1:] {
		for indexNeuron, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {


				update := n.neuronLayers[indexLayer].deltas[indexNeuron]

				weightUpdated := inputConnection.Weight - n.learningRate * update * inputConnection.From.output

				inputConnection.Weight = weightUpdated
			}
		}
	}
}


func (n *NeuralNetwork) feedForward(trainingSampleInput []float32)[]float32{
	var actual []float32
	for neuronLayerIndex, neuronLayer := range n.neuronLayers {
		if neuronLayerIndex == 0 {
			/**
			 * first layer of neurons just passes through trainingSampleInput
			 */
			for trainingInputIndex, trainingValue := range trainingSampleInput{
				//fmt.Printf("trainingInputIndex:: %v, layer size: %v, training sample sie: %v\n", trainingInputIndex, len(neuronLayer.Neurons) , len(trainingSampleInput))
				neuronLayer.Neurons[trainingInputIndex].output = trainingValue
			}
		} else {
			for _, neuron := range neuronLayer.Neurons {

				/*var wg sync.WaitGroup

				wg.Add(len(neuronLayer.Neurons))

				go func() {
					defer wg.Done()
					n.calculateNeuronOutput(neuron, neuronLayerIndex)
				}()
				*/

				n.calculateNeuronOutput(neuron, neuronLayerIndex)

				// Igor's commments:
				// The goroutines don't work as they should here.
				// Problem 1:
				// var wg sync and wg.Add need to go out of this loop, but stil inside live else {}
				// Then, after the loop is over we can call wg.Wait() and wait for all the neurons
				// to have their output values calculated, and only then we can proceed to next layer
				// Unfortunatelly, the fix of Problem 1 leads to Problem 2
				// Problem 2:
				// Instead of go routine calculatingNeuronOutput for each separate Neuron in Layer, it instead
				// calculates it always with the last neuron of the layer multiple time.
				// I believe we have some sort of lock on n NeuralNetwork here, and only the go routines are
				// not able to access it while we are looping through the neurons.
				// The fix for this is not trivial in my opinion. For now, i would avoid the usage of
				// go routines in this particular segmment

				/**
				 * we came to the end
				 */
				if neuronLayerIndex == len(n.neuronLayers) - 1 {
					actual = append(actual, neuron.output)
					//fmt.Println(strconv.FormatFloat(float64(neuron.output), 'E', -1, 32))
				}
			}
		}

	}
	return actual

}

func (n *NeuralNetwork) calculateNeuronOutput(neuron *Neuron, neuronLayerIndex int){
	propagation := n.propagate(neuron.InputConnections)
	bias := n.neuronLayers[neuronLayerIndex - 1].Bias
	neuron.output = n.activationFunction.Activate(propagation + bias * 1.0)
}
