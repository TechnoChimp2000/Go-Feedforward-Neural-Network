package network

import (
	"math"
	"strconv"
	///"sync"
//	"fmt"


	"fmt"
)


// interfaces
type Callback interface{
	ReceiveInfo(info string)
}

type ActivationFunction interface{
	Activate (input float32) float32
	Derivative(input float32) float32
	min() float32
	max() float32
}

type LogisticActivationFunction struct{}

func (l *LogisticActivationFunction) Activate (input float32) float32 {
	return float32(1.0/(1.0+math.Exp(-float64(input))))
}

func (l *LogisticActivationFunction) Derivative (input float32) float32 {
	return input * (1 - input)
}

func (l *LogisticActivationFunction) min() float32 {
	return 0.0
}

func (l *LogisticActivationFunction) max() float32 {
	return 1.0
}


// structures
type Connection struct{
	From          *Neuron
	To            *Neuron
	Weight        float32
}


type Neuron struct {
	// data
	output                 float32
	InputConnections       []*Connection
	ConnectedToInNextLayer []*Neuron
}

type NeuronLayer struct{
	Neurons []*Neuron
	layer   uint8
	Bias    float32
	NumberOfInputConnections int
}


type TrainingSample struct{
	Input  []float32
	Output []float32
}


type NeuralNetwork struct{
	NeuronLayers       []*NeuronLayer
	LearningRate       float32
	TrainingSet        []TrainingSample

	//precision tells how precise network should  be
	//that is error should be lesser than precision
	//typical value is 0.05
	Precision          float32


	// functions
	ActivationFunction ActivationFunction

}

func (w *NeuralNetwork) propagate(inputConnections []*Connection) float32{
	var result float32 = 0.0
	for _,inputConnection := range inputConnections{
		result += inputConnection.From.output * inputConnection.Weight
		//fmt.Println(inputConnection.From.output, inputConnection.Weight)
	}

	return result
}

func (n NeuralNetwork) TrainOffline( iterations int ) {
	// has to go through all the samples, do an update, evaluate, repeat until a certain condition is met.
	// Conditions can be -- precision, number of repeats,

	// VALIDATE whether training samples are defined

	// ITERATIONS
	for i := 0; i < iterations; i++ {


		var errorTotal, regularizationTerm float32 // TODO: regularizationTerm will stay at 0 for now, but we'll add it later

		for _, trainingSample := range n.TrainingSet {

			// FEED FORWARD
			actual := n.FeedForward(trainingSample.Input)
			_error := n.calculateTotalError(actual, trainingSample.Output)

			errorTotal += _error
			//fmt.Printf("AFTER: error total: %v\n", errorTotal)

			errorTotal = errorTotal / float32(len(n.TrainingSet)) + regularizationTerm

		}

		// BACK PROPAGATION
		var deltas map[int][]float32
		for _, trainingSample := range n.TrainingSet {
				deltas = n.backPropagate(trainingSample.Output)
		}
		n.updateWeightsFromDeltas(deltas)
	}
}

func (n *NeuralNetwork) TrainOnline(callback Callback){
	//TODO validate input
	totalSamplesTrained := 0
	for {
		for trainingIndex, trainingSample := range n.TrainingSet {

			if trainingIndex == 0 {
				totalSamplesTrained = 0;
			}
			actual := n.FeedForward(trainingSample.Input)
			_error := n.calculateTotalError(actual, trainingSample.Output)


			if _error < n.Precision {
				totalSamplesTrained++
			}else{
				n.backPropagate(trainingSample.Output)
			}

			if callback != nil {
				callback.ReceiveInfo("For training sample with index: " + strconv.Itoa(trainingIndex) +
				" error is: " + strconv.FormatFloat(float64(_error), 'E', -1, 32) + ". Total samples trained are: " + strconv.Itoa(totalSamplesTrained))
			}

		}
		if totalSamplesTrained == len(n.TrainingSet){
			break;
		}
	}


}

func (n *NeuralNetwork) backPropagate(trainingSampleOutput []float32) (deltas map[int][]float32) {  // deltas[indexLayer][indexNeuron]

	deltas	= make(map[int][]float32)

	for i := len(n.NeuronLayers) - 1; i > 0; i-- {
		/**
		 * processing last layer of neuron connections
		 */
		if(i == len(n.NeuronLayers) - 1){

			/*var wg sync.WaitGroup

			wg.Add(n.NeuronLayers[i].NumberOfInputConnections)*/

			for neuronIndex, neuron := range n.NeuronLayers[i].Neurons {

				// First calculate the last layer deltas
				deltas[i] = append(deltas[i], neuron.output - trainingSampleOutput[neuronIndex] ) // map[2:[0.7413651 -0.21707153]]

				for range neuron.InputConnections {
					//TODO following 5 lines could occur in goroutine

/*					go func() {
						defer wg.Done()*/

					factor1 := deltas[i][neuronIndex]
					factor2 := n.ActivationFunction.Derivative(neuron.output)
					deltas[i-1] = append(deltas[i-1] , factor1 * factor2) //deltasV2: map[2:[0.7413651 -0.21707153] 1:[0.13849856 0.13849856 -0.038098235 -0.038098235]]:

					// l(N-1) :: delta(N-1) = delta(N) * Derivative(N-1) * Output(N-1)  	// factor1 * factor2 * factor3 // we need to store this somewhere so that we can keep on using it,
					// deltas[weightIndex] --> gradients for this index. Update rule is then weight - (alpha * gradient)
					// l(N-2) :: delta(N-2) = delta(N-1) * Derivative(N-2) * Output(N-2)
					// now we have delta(N-1) values and it's time to move to the next layer
				}
			}
		}else{

			for neuronIndex, neuron := range n.NeuronLayers[i].Neurons {
				for range neuron.InputConnections {

					//TODO following lines could occur in goroutine
					var factor1 float32 = 0.0

					for neuronInNextLayerIndex, neuronInNextLayer := range neuron.ConnectedToInNextLayer {

						var weight float32 = neuronInNextLayer.InputConnections[neuronIndex].Weight

						factor1 += deltas[i][2*neuronInNextLayerIndex] * weight // what do we want here? 0.74*0.18 * w5 OR delta * w
					}

					factor2 := n.ActivationFunction.Derivative(neuron.output)
					deltas[i-1] = append(deltas[i-1], factor1*factor2)
				}
			}
		}
	}
	return deltas
}

func (n *NeuralNetwork) updateWeightsFromDeltas(deltas map[int][]float32 ) {

	for indexLayer, layer := range n.NeuronLayers[1:] {
		for indexNeuron, neuron := range layer.Neurons {
			for indexConnection, inputConnection := range neuron.InputConnections {
				weightUpdated := inputConnection.Weight - n.LearningRate * deltas[indexLayer][indexNeuron * len(layer.Neurons)+indexConnection]*inputConnection.From.output
				//fmt.Printf("Original weight:: %v, Updated weight: %v\n", inputConnection.Weight, weightUpdated)
				inputConnection.Weight = weightUpdated
			}
		}
	}
}

// TODO: This function is no longer needed. Can be deleted
func (n *NeuralNetwork) updateWeights(weights []float32) {
	// order of weights: w1, w2, w3, w4, ...
	weightIndex := 0

	for _, layer := range n.NeuronLayers[1:] {
		for _, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {
				fmt.Printf("Original weight:: %v, Updated weight: %v\n", inputConnection.Weight, weights[ weightIndex] )
				//inputConnection.Weight = weights[ weightIndex ]
				weightIndex++
			}
		}
	}
}

// TODO: this function is no longer needed. Can be deleted.
func (n *NeuralNetwork) calculateModifiedWeightOnLastConnectionLayer(neuron *Neuron, inputConnection *Connection, target float32) ( weight_updated float32 )   {
	var factor1 float32 = neuron.output - target
	var factor2 float32 = n.ActivationFunction.Derivative(neuron.output)
	var factor3 float32 = inputConnection.From.output

	var gradient float32 = factor1 * factor2 * factor3
	weight_updated = inputConnection.Weight - (n.LearningRate * gradient)

	return weight_updated
}

func (n *NeuralNetwork) calculateTotalError(actual []float32, output []float32) float32{
	var retval float32 = 0.0
	for outputIndex, singleOutput := range output{
		factor := singleOutput - actual[outputIndex]
		retval += 0.5 * factor * factor
	}
	return retval
}

func (n *NeuralNetwork) FeedForward(trainingSampleInput []float32)[]float32{
	var actual []float32
	for neuronLayerIndex, neuronLayer := range n.NeuronLayers {
		if neuronLayerIndex == 0 {
			/**
			 * first layer of neurons just passes through trainingSampleInput
			 */
			for trainingInputIndex, trainingValue := range trainingSampleInput{
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
				if neuronLayerIndex == len(n.NeuronLayers) - 1 {
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
	bias := n.NeuronLayers[neuronLayerIndex - 1].Bias
	neuron.output = n.ActivationFunction.Activate(propagation + bias * 1.0)
}



