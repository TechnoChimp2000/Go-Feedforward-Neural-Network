package network

import (
	"strconv"
	"fmt"
	"reflect"
)

type Trainer interface{
	train(n *NeuralNetwork, trainingSet []TrainingSample)
}

type OnlineTrainer struct{}

func (o *OnlineTrainer) train(n *NeuralNetwork, trainingSet []TrainingSample){
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

type OfflineTrainer struct{
	BatchSize, Epoch int
}

//TODO: Here we have 2 functions that access two exported fields of the OfflineTrainer struct, but they are defined as methods of NeuralNetwork ( which can have other trainers as well)
//TODO: Therefore, we need to make this more generic -- first time I used the reflect package, so this might not be the most optimal way of doing it
func (n *NeuralNetwork) SetOfflineEpoch(epoch int64) {

	x := reflect.ValueOf(n.trainer).Elem().FieldByName("Epoch")
	x.SetInt(epoch)

}

func (n *NeuralNetwork) SetOfflineBatchSize(batchSize int64) {

	x := reflect.ValueOf(n.trainer).Elem().FieldByName("BatchSize")
	x.SetInt(batchSize)

}

func (o *OfflineTrainer) train( n *NeuralNetwork, trainingSet []TrainingSample){

	// we need to loop through all the training samples, collect all the gradients and then update the weights
	if n.debug {
		fmt.Printf("Offline trainer start.\n")
		fmt.Printf("epoch: %v, batchsize: %v\n", o.Epoch, o.BatchSize)
	}

	var _error, batchCount float32

	// create the delta accumulator
	deltaAccumulator := make([][]float32, len(n.neuronLayers)-1)
	for i, layer := range n.neuronLayers[1:] {
		deltaAccumulator[i] = make([]float32, len(layer.Neurons))
	}

	// do n iterations - passed to this by the function arguments
	for i:=0; i < o.Epoch; i++ {

		j := 0 // mini batch
		for _, currentTrainingSample := range trainingSet {

			// feedForward gets us Neuron outputs as well
			actual 	:= n.feedForward(currentTrainingSample.Input)
			_error 	+= n.costFunction.calculateTotalError(actual, currentTrainingSample.Output)

			// backPropagate changes to deltas[]float32 in NeuronLayers
			n.backPropagate(currentTrainingSample.Output)

			// storing deltas in the deltaAccumulator
			for indexLayer, layer := range n.neuronLayers[1:] {
				for indexNeuron, neuron := range layer.Neurons {
					// accumulates the initial value plus the newly backpropagated 'delta' value
					deltaAccumulator[indexLayer][indexNeuron] += neuron.Delta

				}
			}

			// mini batch counter j
			j++
			if (j == o.BatchSize ) {
				batchCount++

				// Take the average of each delta by dividing it with the batchSize
				// if batchSize equals the length of training samples then we effectively have an offline trainer, or normal gradient descent

				for indexLayer, layer := range n.neuronLayers[1:] {
					for indexNeuron, neuron := range layer.Neurons {
						neuron.Delta = deltaAccumulator[indexLayer][indexNeuron] / float32( o.BatchSize )
					}
				}

				if n.debug {
					fmt.Printf("Epoch: %v, Batch number: %v Error: %v\n", i, batchCount, _error)
				}


				n.updateWeightsFromDeltas()

				// reset the delta accumulators to 0
				for indexLayer := range deltaAccumulator {
					for indexNeuron := range deltaAccumulator[indexLayer] {
						deltaAccumulator[indexLayer][indexNeuron] = 0
					}
				}

				// reset the total error and j
				_error	= 0
				j	= 0
			}
		}
	}
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

func (n *NeuralNetwork) backPropagate(trainingSampleOutput []float32) {

	for indexLayer := len(n.neuronLayers) - 1; indexLayer > 0; indexLayer-- {

		// last layer aka output layer
		if(indexLayer == len(n.neuronLayers) - 1){

			for indexNeuron, neuron := range n.neuronLayers[indexLayer].Neurons {
				n.costFunction.calculateWeightDeltaInLastLayer(n, neuron, indexNeuron, indexLayer, trainingSampleOutput)

			}

		} /* other layers */else {
			for indexNeuron, neuron := range n.neuronLayers[indexLayer].Neurons {
				n.costFunction.calculateWeightDelta(n, neuron, indexNeuron, indexLayer, trainingSampleOutput)
			}
		}
	}
}

func (n *NeuralNetwork) updateWeightsFromDeltas() {

	for _, layer := range n.neuronLayers[1:] {
		for _, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {

				update := neuron.Delta
				//update := n.neuronLayers[indexLayer].deltas[indexNeuron]

				weightUpdated := inputConnection.Weight - n.learningRate * update * inputConnection.From.output

				inputConnection.Weight = weightUpdated


			}
			// update bias

			neuron.Bias -= n.learningRate * neuron.Delta
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
	bias := neuron.Bias
	neuron.output = n.activationFunction.Activate(propagation + bias * 1.0)
}
