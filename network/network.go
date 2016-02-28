package network

import "math"


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
	return (*l).Activate(input) * (1.0 - (*l).Activate(input))
}

func (l *LogisticActivationFunction) min() float32 {
	return 0.0
}

func (l *LogisticActivationFunction) max() float32 {
	return 1.0
}


// structures
type Connection struct{
	from Neuron
	to Neuron
	weight float32
	updatedWeight float32

}


type Neuron struct {
	// data
	output float32

	inputConnections []*Connection
	connectedToInNextLayer []Neuron
	//outputConnections []*Connection
}

type NeuronLayer struct{
	neurons []Neuron
	layer uint8
	bias float32
}


type TrainingSample struct{
	input []float32
	output []float32
}


type NeuralNetwork struct{
	neuronLayers []*NeuronLayer
	learningRate float32
	trainingSet []TrainingSample

	//precision tells how precise network should  be
	//that is error should be lesser than precision
	//typical value is 0.05
	precision float32


	// functions
	activationFunction ActivationFunction

}

func (w *NeuralNetwork) propagate(inputConnections []*Connection) float32{
	var result float32 = 0.0
	for _,inputConnection := range inputConnections{
		result += inputConnection.from.output + inputConnection.weight
	}
	return result
}

func (n NeuralNetwork) TrainOffline(){

}


func (n *NeuralNetwork) TrainOnline(callback Callback){
	//TODO validate input
	totalSamplesTrained := 0
	for {
		for trainingIndex, trainingSample := range n.trainingSet {

			if trainingIndex == 0 {
				totalSamplesTrained = 0;
			}
			actual := n.feedForward(trainingSample.input)
			_error := n.calculateTotalError(actual, trainingSample.output)


			if _error < n.precision {
				totalSamplesTrained++
			}else{
				n.backPropagate(trainingSample.output)
			}

			if callback != nil {
				//callback.ReceiveInfo("For training sample with index: " + trainingIndex + " error is: " + _error + ". Total samples trained are: " + totalSamplesTrained)
			}

		}
		if totalSamplesTrained == len(n.trainingSet){
			break;
		}
	}


}

func (n *NeuralNetwork) backPropagate(trainingSampleOutput []float32){
	for i := len(n.neuronLayers) - 1; i > 0; i-- {
		/**
		 * processing last layer of neuron connections
		 */
		if(i == len(n.neuronLayers) - 1){
			for neuronIndex, neuron := range n.neuronLayers[i].neurons{
				for _, inputConnection := range neuron.inputConnections{

					//TODO following 5 line could occur in goroutine
					var factor1 float32 = neuron.output - trainingSampleOutput[neuronIndex]
					var factor2 float32 = n.activationFunction.Derivative(neuron.output)
					var factor3 float32 = inputConnection.from.output
					var gradient float32 = factor1 * factor2 * factor3
					inputConnection.updatedWeight = inputConnection.weight - (n.learningRate * gradient)
				}

			}
		}else{
			for _, neuron := range n.neuronLayers[i].neurons{
				for _, inputConnection := range neuron.inputConnections{

					//TODO following lines could occur in goroutine
					var factor1 float32 = 0.0
					for _, neuronInNextLayer := range neuron.connectedToInNextLayer{
						//TODO factor11 and factor12 have already been calculated in previous  steps
						var factor11 float32 = n.activationFunction.Derivative(neuronInNextLayer.output)
						var factor12 float32 = neuron.output
						factor1 += factor11 * factor12
					}

					factor2 := n.activationFunction.Derivative(neuron.output)
					factor3 := inputConnection.from.output
					gradient := factor1 * factor2 * factor3
					inputConnection.updatedWeight = inputConnection.weight - (n.learningRate * gradient)
				}

			}
		}
	}

	/**
	 * finally update weights
	 */
	for neuronLayerIndex, neuronLayer := range n.neuronLayers{
		if neuronLayerIndex == 0 {
			continue
		}
		for _, neuron := range  neuronLayer.neurons{
			for _, inputConnection := range neuron.inputConnections{
				inputConnection.weight = inputConnection.updatedWeight
			}
		}

	}


}

func (n *NeuralNetwork) calculateTotalError(actual []float32, output []float32) float32{
	var _error float32 = 0.0
	for outputIndex, singleOutput := range output{
		factor := singleOutput - actual[outputIndex]
		_error += 0.5 *factor * factor
	}
	return _error
}

func (n *NeuralNetwork) feedForward(trainingSampleInput []float32)[]float32{
	var actual []float32
	for neuronLayerIndex, neuronLayer := range n.neuronLayers{
		if neuronLayerIndex == 0 {
			/**
			 * first layer of neurons just passes through trainingSampleInput
			 */
			for trainingInputIndex, trainingValue := range trainingSampleInput{
				neuronLayer.neurons[trainingInputIndex].output = trainingValue
			}
		} else {
			for _, neuron := range neuronLayer.neurons{
				//TODO following three lines could occur in goroutines
				propagation := n.propagate(neuron.inputConnections)
				bias := n.neuronLayers[neuronLayerIndex - 1].bias
				neuron.output = n.activationFunction.Activate(propagation + bias * 1.0)
			}
		}

		/**
		 * we came to the end
		 */
		if neuronLayerIndex == len(n.neuronLayers) - 1 {
			for _, neuron := range neuronLayer.neurons{
				actual = append(actual, neuron.output)
			}
		}

	}
	return actual

}



