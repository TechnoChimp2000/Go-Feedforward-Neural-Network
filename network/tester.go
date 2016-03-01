package network

import (
	"fmt"
)

// a set of Methods designed to test the various methods of the neural Network

// Functions
/*
DONE:::::LogisticActivationFunction.Activate
DONE:::::LogisticActivationFunction.Derivative
DONE:::::PropagationWithoutBias - takes 4 input layer neurons, and propagates them into 3 hidden layer neurons
DONE:::::PropagationWithBias

Partially Done: FeedForward ( i think that after the goroutines have been removed, it works as it should.

TODO: Complete FeedForward by hand, and see if the values you get are comparable to the ones in the output
TODO: Continue with writing tests for functions that still need to be tested

Neuron ( InputConnections []*Connection, ConnectedToInNextLayer []*Neuron ) (output)
Connection ( From *Neuron, To *Neuron, weight float)
NeuronLayer ( Neurons[]*Neuron, layer uint8, Bias float32, NumberOfInputConnections int )
NeuralNetwork (NeuronLayars []*NeuronLayer, LearningRate float32, TrainingSet []TrainingSample, precision float32, ActivationFunction ActivationFunction)

Forward Propagation test
	(* NeuralNetwork) propagate(inputConnections []*Connection) float32
	(* NeuralNetwork) FeedForward(trainingSampleInput []float32) float32

	(* NeuralNetwork) calculateNeuronOutput (neuron *Neuron, neuraonLayerIndex int)

Backward Propagation test
	(* NeuralNetwork) backPropagate(trainingSampleOutput []float32)

TrainTest
	(* NeuralNetwork) TrainOnline(callback Callback)

	(* NeuralNetwork) calculateModifiedWeightOnLastConnectionLayer (neuron *Neuron, inputConnection *Connection, targer float32)

	(* NeuralNetwork) calculateTotalError(actual []float32, output []float32)
*/

type Test struct {
	Name		string
	Success		bool
	Input		float32
	Output		float32
	Expected_output float32
}


func StartTesting() (tests []Test) {

	fmt.Println("Testing of functions ... ")

	// logistic activation function
	var LAF LogisticActivationFunction
	// Activate
	var LAFActivate Test

	LAFActivate.Success 	= LAF.runActivate()
	LAFActivate.Name	= "LogisticActivationFunction.Activate"

	tests = append(tests, LAFActivate)
	// Derivative
	var LAFDerivative Test

	LAFDerivative.Success	= LAF.runDerivative()
	LAFDerivative.Name	= "LogisticActivationFunction.Derivative"

	tests = append(tests, LAFDerivative)
	///////////////////////////////////

	// Various Network Tests

	// create a simple network with 4-3-3; one input layer, one hidden layer with a bias unit and an output layer

	neuronNetwork := CreateSimpleNetwork()

	// Forward Propagation Testing

	// WEIGHTS
	// initialize weights aka connections. Let's loop through all neurons in l1 and l2
	neuronNetwork.InitializeWeights()

	// INITIALIZE INPUT CONNECTIONS FOR layer1 AND layer2
	neuronNetwork.InitializeInputConnections()

	// CHANGE OUTPUT of Neurons in input_layer to be equal to
	neuronNetwork.InsertOneTrainingSample()

	var NNpropagateSimple Test

	NNpropagateSimple.Success	= neuronNetwork.runSimplePropagateTest()
	NNpropagateSimple.Name		= "PropagationWithoutBias"

	tests = append(tests, NNpropagateSimple)

	var NNpropagateWithBias Test

	NNpropagateWithBias.Success	= neuronNetwork.runPropagationWithBias()
	NNpropagateWithBias.Name	= "PropagationWithBias"

	tests = append(tests, NNpropagateWithBias)

	// FULL FEED FORWARD TEST

	var FeedForward Test

	FeedForward.Success	= neuronNetwork.runFeedForward()
	FeedForward.Name	= "FeedForward"
	return tests
}

func (n *NeuralNetwork) runFeedForward() (success bool) {

	output := n.FeedForward(n.TrainingSet[0].Input)
	//fmt.Println(n.TrainingSet[0].Input) // outputs 1 2 3 4 -- which is correct
	// 1 - training values are correctly set as the output of input neurons

	fmt.Println(output) // VALUES feed forward gives me: [0.9830435 0.99999785 1]



	return success
}

// Since we'll be 'faking' layer1 outputs, the argument should not be a pointer to our neural network
func (n NeuralNetwork) runPropagationWithBias() (success bool) {

	layer1 := n.NeuronLayers[1]
	layer2 := n.NeuronLayers[2]

	bias 			:= layer1.Bias
	layer1.Neurons[0].output = 1
	layer1.Neurons[1].output = 2
	layer1.Neurons[2].output = 3

	// our output[weight] pairs and --> results are
	/*
		1 0.02
		2 1.02
		3 2.02
		RESULT: 8.62

		1 3.02
		2 4.02
		3 5.02
		RESULT: 26.619999

		1 6.02
		2 7.02
		3 8.02
		RESULT: 44.620003
	*/
	//
	real_values := []float32{ 8.62, 26.619999, 44.620003 }

	for i, neuronInLayer2 := range layer2.Neurons {
		//fmt.Println("input connection for neuron:",i, " in layer1: ", neuronInLayer2.InputConnections)
		// first sum up bias with a hard-coded weight
		propagateValue := bias * 0.5
		// propagate
		propagateValue += n.propagate( neuronInLayer2.InputConnections )

		if propagateValue == real_values[i] {
			success = true
		} else {
			success = false
		}
	}
	return success
}

func (n *NeuralNetwork) runSimplePropagateTest() (success bool) {

	layer1 := n.NeuronLayers[1]

	// our output[weight] pairs and --> results are
	/*
	1 0.01
	2 1.01
	3 2.01
	4 3.01
	RESULT: 20.099998 (excel value is 20.1)

	1 4.01
	2 5.01
	3 6.01
	4 7.01
	RESULT: 60.100002

	1 8.01
	2 9.01
	3 10.01
	4 11.01
	RESULT: 100.100006
	*/
	//
	real_values := []float32{20.099998, 60.100002, 100.100006}

	for i, neuronInLayer1 := range layer1.Neurons {
		//fmt.Println("input connection for neuron:",i, " in layer1: ", neuronInLayer1.InputConnections)

		// propagate
		propagateValue := n.propagate( neuronInLayer1.InputConnections )

		if propagateValue == real_values[i] {
			success = true
		} else {
			success = false
		}
		//fmt.Println(propagateValue)
	}
	return success
}

func (n *NeuralNetwork) InsertOneTrainingSample() {

	input_layer := n.NeuronLayers[0]
	input_layer.Neurons[0].output = n.TrainingSet[0].Input[0]
	input_layer.Neurons[1].output = n.TrainingSet[0].Input[1]
	input_layer.Neurons[2].output = n.TrainingSet[0].Input[2]
	input_layer.Neurons[3].output = n.TrainingSet[0].Input[3]
}


func (n *NeuralNetwork) InitializeInputConnections() {

	input_layer := n.NeuronLayers[0]
	layer1 := n.NeuronLayers[1]
	layer2 := n.NeuronLayers[2]

	// LAYER 1
	for i, neuronInLayer1 := range layer1.Neurons {
		for j, _ := range input_layer.Neurons {
			//			fmt.Println(i, j)
			//			fmt.Println("weights position", ( i* len(layer1.Neurons) + j) )
			neuronInLayer1.InputConnections = append( neuronInLayer1.InputConnections, weights[i*len(input_layer.Neurons)+j] )
		}
	}

	//fmt.Println("weights from input_layer to layer1;  ")
	/*
	for _ , neuron := range layer1.Neurons {
		for j, _ := range neuron.InputConnections {
			//fmt.Print(neuron.InputConnections[j].Weight, " ")
			//fmt.Print(neuron.InputConnections[j].From.output, " ")
		}
	}
	*/
	//fmt.Print("\n")
	//fmt.Println(len(weights[0:12])) 12 elemtns, w[11] is last one
	//fmt.Println(len(weights[12:]))// 9 elements, w[20] is last one


	// LAYER 2
	for i, neuronInLayer2 := range layer2.Neurons {
		// deal with bias first -- TODO: If you don't deal with bias here, you will need to deal with it at propagation stage ...
		for j, _ := range layer1.Neurons {
			neuronInLayer2.InputConnections = append(neuronInLayer2.InputConnections, weights[ layer2.NumberOfInputConnections + i*len(layer1.Neurons) +j ]  )
		}
		//fmt.Println(i, neuronInLayer2.InputConnections)
	}
	/*
	//fmt.Println("weights from layer1 to layer2")
	for _ , neuron := range layer2.Neurons {
		for j, _ := range neuron.InputConnections {
			//fmt.Print(neuron.InputConnections[j].Weight, " ")
		}
	}
	//fmt.Print("\n") */
}

var weights []*Connection

func (n *NeuralNetwork) InitializeWeights() {

	w_l1 := float32(0.01)
	w_l2 := float32(0.02)

	input_layer 	:= n.NeuronLayers[0]
	layer1		:= n.NeuronLayers[1]
	layer2 		:= n.NeuronLayers[2]


	for _, neuronInLayer1 := range layer1.Neurons {
		for _, inputNeuron := range input_layer.Neurons {
			w := &Connection{From: inputNeuron, To: neuronInLayer1, Weight: w_l1}
			weights = append(weights, w)
			w_l1 += 1
		}
	}

	for _, neuronInLayer2 := range layer2.Neurons {
		for _, neuronInLayer1 := range layer1.Neurons {
			w := &Connection{From: neuronInLayer1, To: neuronInLayer2, Weight: w_l2}
			weights = append(weights, w)
			w_l2 += 1
		}
	}

	//fmt.Println("length of weights: ", len(weights)) //length of weights:  21. It is correct. We have 4*3 = 12 weights in the inputlayer to layer1 interaction, and 3*3 weights in layer1 to layer2 interaction
	/*
	fmt.Println(" all weights. .01 are from inputl to l1, and .02 are from l1 to l2")
	for _, w := range weights {
		fmt.Print(w.Weight, " ")
	}
	fmt.Print("\n")
	*/
}

func CreateSimpleNetwork() (n *NeuralNetwork) {

	// build the input layer
	input_layer := CreateNeuronLayer(4,0,0)

	// build two layers of neurons
	layer1	:= CreateNeuronLayer(3, 12, 1 ) // Definition of 2nd parameter (number of input connections): (Bias + neurons in previous layer ) * neurons in this layer
	layer2	:= CreateNeuronLayer(3, 12, 0 ) // this is essentially the output layer in my case


	// connect them into 3 layers.
	var neuronLayers []*NeuronLayer

	neuronLayers = append(neuronLayers, & input_layer)
	neuronLayers = append(neuronLayers, & layer1)
	neuronLayers = append(neuronLayers, & layer2)

	// define connections for neurons from input_layer to layer1
	for _, inputNeuron := range input_layer.Neurons {
		for _, neuronInLayer1 := range layer1.Neurons {
			inputNeuron.ConnectedToInNextLayer = append( inputNeuron.ConnectedToInNextLayer, neuronInLayer1 )
		}
	}

	// define connections for neurons from layer1 to neurons in layer2
	for _,neuronInLayer1 := range layer1.Neurons {
		for _, neuronInLayer2 := range layer2.Neurons {
			neuronInLayer1.ConnectedToInNextLayer = append(neuronInLayer1.ConnectedToInNextLayer, neuronInLayer2)
		}
	}

	// Create one training sample for the network and turn it into a 1 element slice
	ts := TrainingSample{Input:[]float32{1.00, 2.00,3.00,4.00}, Output:[]float32{1.00,0.00,0.00}}
	TrainingSamples := []TrainingSample{ts}

	// declare neural network
	neuronNetwork := NeuralNetwork{NeuronLayers: neuronLayers, TrainingSet:TrainingSamples, LearningRate: 0.02, Precision:0.0003, ActivationFunction: new(LogisticActivationFunction)}
	// TODO: Remove the TrainingSet from creation of network. It is not needed at this point and can be done later with other functions.

	return &neuronNetwork
}


func (l *LogisticActivationFunction) runActivate() (success bool) {

	test_values := []float32{-10000.0000, -2, 0, 2, 10000.0000}
	real_values := []float32{0, 0.11920292, 0.5, 0.8807971, 1 }

	for j, i := range test_values {

		result := l.Activate(i)

		if result == real_values[j] {
			success = true
		} else {
			success = false
			return success
		}
	}
	return success
}

func (l *LogisticActivationFunction) runDerivative() (success bool) {

	test_values := []float32{-10000.0000, -2, 0, 2, 10000.0000}
	real_values := []float32{0, 0.10499358, 0.25, 0.104993574, 0 }

	for j, i := range test_values {

		g 		:= l.Activate(i)
		gradient	:= l.Derivative(g)

		if gradient == real_values[j] {
			success = true
		} else {
			success = false
			return success
		}
	}
	return success
}

