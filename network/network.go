package network

import (
	"math"
        "strconv"
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
	From          Neuron
	To            Neuron
	Weight        float32
	updatedWeight float32

}


type Neuron struct {
	// data
	output                 float32

	InputConnections       []*Connection
	ConnectedToInNextLayer []Neuron
	//outputConnections []*Connection
}

type NeuronLayer struct{
	Neurons []Neuron
	layer   uint8
	Bias    float32
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
		result += inputConnection.From.output + inputConnection.Weight
	}
	return result
}

func (n NeuralNetwork) TrainOffline(){

}


func (n *NeuralNetwork) TrainOnline(callback Callback){
	//TODO validate input
	totalSamplesTrained := 0
	for {
		for trainingIndex, trainingSample := range n.TrainingSet {

			if trainingIndex == 0 {
				totalSamplesTrained = 0;
			}
<<<<<<< HEAD
			actualOuput := n.feedForward(trainingSample.input)
			totalError := n.calculateTotalError(actualOuput, trainingSample.output)


			if totalError < n.precision {
=======
			actual := n.feedForward(trainingSample.Input)
			_error := n.calculateTotalError(actual, trainingSample.Output)


			if _error < n.Precision {
>>>>>>> 1afea1827c636814176215fedae8716c5ad31a1a
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

func (n *NeuralNetwork) backPropagate(trainingSampleOutput []float32){
	for i := len(n.NeuronLayers) - 1; i > 0; i-- {
		/**
		 * processing last layer of neuron connections
		 */
		if(i == len(n.NeuronLayers) - 1){
			for neuronIndex, neuron := range n.NeuronLayers[i].Neurons {
				for _, inputConnection := range neuron.InputConnections {

					//TODO following 5 line could occur in goroutine
					var factor1 float32 = neuron.output - trainingSampleOutput[neuronIndex]
					var factor2 float32 = n.ActivationFunction.Derivative(neuron.output)
					var factor3 float32 = inputConnection.From.output
					var gradient float32 = factor1 * factor2 * factor3
					inputConnection.updatedWeight = inputConnection.Weight - (n.LearningRate * gradient)
				}

			}
		}else{
			for _, neuron := range n.NeuronLayers[i].Neurons {
				for _, inputConnection := range neuron.InputConnections {

					//TODO following lines could occur in goroutine
					var factor1 float32 = 0.0
					for _, neuronInNextLayer := range neuron.ConnectedToInNextLayer {
						//TODO factor11 and factor12 have already been calculated in previous  steps
						var factor11 float32 = n.ActivationFunction.Derivative(neuronInNextLayer.output)
						var factor12 float32 = neuron.output
						factor1 += factor11 * factor12
					}

					factor2 := n.ActivationFunction.Derivative(neuron.output)
					factor3 := inputConnection.From.output
					gradient := factor1 * factor2 * factor3
					inputConnection.updatedWeight = inputConnection.Weight - (n.LearningRate * gradient)
				}

			}
		}
	}

	/**
	 * finally update weights
	 */
	for neuronLayerIndex, neuronLayer := range n.NeuronLayers {
		if neuronLayerIndex == 0 {
			continue
		}
		for _, neuron := range  neuronLayer.Neurons {
			for _, inputConnection := range neuron.InputConnections {
				inputConnection.Weight = inputConnection.updatedWeight
			}
		}

	}


}

func (n *NeuralNetwork) calculateTotalError(actual []float32, output []float32) float32{
	var retval float32 = 0.0
	for outputIndex, singleOutput := range output{
		factor := singleOutput - actual[outputIndex]
		retval += 0.5 *factor * factor
	}
	return retval
}

func (n *NeuralNetwork) feedForward(trainingSampleInput []float32)[]float32{
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
				//TODO following three lines could occur in goroutines
				propagation := n.propagate(neuron.InputConnections)
				bias := n.NeuronLayers[neuronLayerIndex - 1].Bias
				neuron.output = n.ActivationFunction.Activate(propagation + bias * 1.0)
			}
		}

		/**
		 * we came to the end
		 */
		if neuronLayerIndex == len(n.NeuronLayers) - 1 {
			for _, neuron := range neuronLayer.Neurons {
				actual = append(actual, neuron.output)
			}
		}

	}
	return actual

}



