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
DONE:::::FeedForward ( i think that after the goroutines have been removed, it works as it should.
DONE:::::BackPropagation
DONE:::::TrainOffline

*/

type Test struct {
	Name		string
	Success		bool
	Input		float32
	Output		[]float32
	Expected_output float32
	Comment		string
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
	weights := neuronNetwork.InitializeTestWeights()

	// INITIALIZE INPUT CONNECTIONS FOR layer1 AND layer2
	neuronNetwork.InitializeInputConnections(weights)

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

	tests = append(tests, FeedForward)

	// BACK PROPAGATION TESTS
	var BackPropagation Test

	BackPropagation.Success, BackPropagation.Output	= neuronNetwork.runBackPropagation()
	BackPropagation.Name				= "BackPropagation"
	BackPropagation.Comment				= "Same results as http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/"

	tests = append(tests, BackPropagation)

	var TrainOffline Test

	neuronNetwork2 := CreateSimpleNetwork()
	weights2 := neuronNetwork2.InitializeTestWeights()
	neuronNetwork2.InitializeInputConnections( weights2 )
	neuronNetwork2.InsertOneTrainingSample()

	TrainOffline.Success				= neuronNetwork2.runTrainOffline()
	TrainOffline.Name				= "TrainOffline"

	tests = append(tests, TrainOffline)

	return tests
}

func (n * NeuralNetwork) runTrainOffline() (success bool) {

	real_weights := []float32{
		0.37851045,
		0.6570201,
		0.4773005,
		0.75460017,
		-3.892975,
		-3.8684173,
		2.8618135,
		2.925688,
		0.37851417,
		0.65702754,
		0.47730425,
		0.7546077,
		-3.8930032,
		-3.8684456,
		2.8618417,
		2.9257164,
	}


	//fmt.Println("weight: ", n.NeuronLayers[1].Neurons[0].InputConnections[0].Weight)
	//n.TrainOffline(10000)
	TrainOffline(10000, n)

	i := 0
	for _, layer := range n.NeuronLayers {
		for _, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {
				//fmt.Println(inputConnection.Weight, real_weights[i])
				if real_weights[i] == inputConnection.Weight {
					success = true
				} else {
					success = false
				}
				i++

			}
		}
	}

	fmt.Println(n.calculateTotalError( n.feedForward(n.TrainingSet[0].Input), n.TrainingSet[0].Output ))


	return success
}

func (n * NeuralNetwork) runBackPropagation() (success bool, output []float32) {
	// Let's do a sample back propagation test with 1 element only!

	sample_output := n.TrainingSet[0].Output

	// get the total number of weights
	weights_length := 0;
	for _, l := range n.NeuronLayers[1:] {
		for _, neuron := range l.Neurons {
			weights_length += len(neuron.InputConnections)
		}
	}

	var actual []float32
	var error float32

	for i := 0; i<10000; i++ {
		actual = n.feedForward(n.TrainingSet[0].Input)
		error = n.calculateTotalError( actual, n.TrainingSet[0].Output )

		//fmt.Printf("Total Error at %v iteration: %v\n",i, error)
		//fmt.Printf("RealOutput Value: %v\n",n.TrainingSet[0].Output)
		//fmt.Printf("FeelForward Value: %v\n",actual)
		//fmt.Printf("TotalError: %v\n", error)

		deltas := n.backPropagate( sample_output )
		n.updateWeightsFromDeltas(deltas)
		//fmt.Println(n.calculateTotalError( actual, n.TrainingSet[0].Output ))
	}

	// final values
	error = n.calculateTotalError( actual, n.TrainingSet[0].Output )
	output = n.feedForward(n.TrainingSet[0].Input)

	fmt.Println("This is the final output:", output )
	fmt.Printf("Sample Output: %v, Final Prediction: %v, Final Error: %v\n", n.TrainingSet[0].Output, output, error) //Sample Output: [0.01 0.99], Final Prediction: [0.015913634 0.9840643], Final Error: 3.510851e-05


	fmt.Println("These are the final weights:")
	for _, layer := range n.NeuronLayers {
		for _, neuron := range layer.Neurons {
			for _, inputConnection := range neuron.InputConnections {
				fmt.Println(inputConnection.Weight)
			}
		}
	}

	real_output := []float32{ 0.015913634, 0.9840643 }

	for i, _ := range real_output {
		if real_output[i] == output[i] {
			success = true
		} else {
			success = false
		}
}
	return success, output
}

func (n *NeuralNetwork) runFeedForward() (success bool) {

	real_values := []float32{ 0.75136507, 0.7729285 }
	output := n.feedForward(n.TrainingSet[0].Input)

	//fmt.Println("Feed Forward output:", output)
	for i, _ := range real_values {

		if output[i] == real_values[i] {
			success = true
		} else {
			success = false
		}
	}
	return success
}

// Since we'll be 'faking' layer1 outputs, the argument should not be a pointer to our neural network
func (n NeuralNetwork) runPropagationWithBias() (success bool) {

	layer1 := n.NeuronLayers[1]
	layer2 := n.NeuronLayers[2]

	bias 			:= layer1.Bias
	layer1.Neurons[0].output = 1
	layer1.Neurons[1].output = 2

	real_values := []float32{ 1.9000001, 2.2 }

	for i, neuronInLayer2 := range layer2.Neurons {
		//fmt.Println("input connection for neuron:",i, " in layer1: ", neuronInLayer2.InputConnections)
		// first sum up bias with a hard-coded weight
		propagateValue := bias * 1
		// propagate
		propagateValue += n.propagate( neuronInLayer2.InputConnections )

		//fmt.Println("propValue: ", propagateValue)
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

	real_values := []float32{0.027500002, 0.0425}

	for i, neuronInLayer1 := range layer1.Neurons {


		// propagate
		propagateValue := n.propagate( neuronInLayer1.InputConnections )

		if propagateValue == real_values[i] {
			success = true
		} else {
			success = false
		}

	}
	return success
}

func (n *NeuralNetwork) InsertOneTrainingSample() {

	input_layer := n.NeuronLayers[0]
	input_layer.Neurons[0].output = n.TrainingSet[0].Input[0]
	input_layer.Neurons[1].output = n.TrainingSet[0].Input[1]

}


func (n *NeuralNetwork) InitializeInputConnections( weights []*Connection) {

	input_layer := n.NeuronLayers[0]
	layer1 := n.NeuronLayers[1]
	layer2 := n.NeuronLayers[2]


	weight_counter := 0 // TODO: weight counter will have to be dealt with 'generically', but not in the test
	// weights length: 8


	// LAYER 1
	for i, neuronInLayer1 := range layer1.Neurons {
		for j, _ := range input_layer.Neurons {
			//fmt.Println("weights position", ( i* len(layer1.Neurons) + j), "weights value:", weights[i+j])
			neuronInLayer1.InputConnections = append( neuronInLayer1.InputConnections, weights[i*len(input_layer.Neurons)+j] )
			weight_counter++
		}
	}


	// LAYER 2
	for i, neuronInLayer2 := range layer2.Neurons {
		// deal with bias first -- TODO: If you don't deal with bias here, you will need to deal with it at propagation stage ...
		for j, _ := range layer1.Neurons {
			//fmt.Println(i + j )
			neuronInLayer2.InputConnections = append(neuronInLayer2.InputConnections, weights[weight_counter+i*len(layer1.Neurons) + j ] )
			//fmt.Println("weights position", ( weight_counter + i*len(layer1.Neurons) + j), "weights value:", weights[weight_counter+i*len(layer1.Neurons) + j ])
		}
	}

}



func (n *NeuralNetwork) InitializeTestWeights() (weights []*Connection) {

	w_l1 := float32(0.15)
	w_l2 := float32(0.40)

	input_layer 	:= n.NeuronLayers[0]
	layer1		:= n.NeuronLayers[1]
	layer2 		:= n.NeuronLayers[2]


	for _, neuronInLayer1 := range layer1.Neurons {
		for _, inputNeuron := range input_layer.Neurons {
			w := &Connection{From: inputNeuron, To: neuronInLayer1, Weight: w_l1}
			weights = append(weights, w)
			w_l1 += 0.05
		}
	}

	for _, neuronInLayer2 := range layer2.Neurons {
		for _, neuronInLayer1 := range layer1.Neurons {
			w := &Connection{From: neuronInLayer1, To: neuronInLayer2, Weight: w_l2}
			weights = append(weights, w)
			w_l2 += 0.05
		}
	}

	return weights
}

func CreateSimpleNetwork() (n *NeuralNetwork) {

	// build the input layer
	input_layer := CreateNeuronLayer(2,0.35)

	// build two layers of neurons
	layer1	:= CreateNeuronLayer(2, 0.6 ) // Definition of 2nd parameter (number of input connections): (Bias + neurons in previous layer ) * neurons in this layer
	layer2	:= CreateNeuronLayer(2, 0 ) // this is essentially the output layer in my case


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
	ts := TrainingSample{Input:[]float32{0.05, 0.1}, Output:[]float32{0.01, 0.99}}
	TrainingSamples := []TrainingSample{ts}

	// declare neural network
	neuronNetwork := NeuralNetwork{NeuronLayers: neuronLayers, TrainingSet:TrainingSamples, LearningRate: 0.5, Precision:0.0003, ActivationFunction: new(LogisticActivationFunction)}
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

func TrainOffline( iterations int, n *NeuralNetwork) (errors []float32) {
	// has to go through all the samples, do an update, evaluate, repeat until a certain condition is met.
	// Conditions can be -- precision, number of repeats,

	// VALIDATE whether training samples are defined

	// ITERATIONS
	for i := 0; i < iterations; i++ {

		var errorTotal, regularizationTerm float32 // TODO: regularizationTerm will stay at 0 for now, but we'll add it later

		for _, trainingSample := range n.TrainingSet {

			// FEED FORWARD
			actual := n.feedForward(trainingSample.Input)
			_error := n.calculateTotalError(actual, trainingSample.Output)

			//fmt.Printf("FeedForward prediction: %v\n", actual)
			errorTotal += _error
			fmt.Println(actual)


			//fmt.Println(_error)
		}

		errorTotal = errorTotal / float32(len(n.TrainingSet)) + regularizationTerm
		fmt.Println("errorTotal:",errorTotal)
		errors = append(errors, errorTotal)


		// BACK PROPAGATION
		var deltas, deltas_buffer map[int][]float32

		for i, trainingSample := range n.TrainingSet {

			if i == 0 {
				deltas = n.backPropagate(trainingSample.Output)
			} else {
				deltas_buffer = n.backPropagate(trainingSample.Output)

				for indexLayer, layer := range deltas_buffer {
					for indexWeight, weight := range layer {
						deltas[indexLayer][indexWeight] += weight

						// if we are in last iteration we need to divide by number of training samples
						if i == ( len(n.TrainingSet) - 1 ) {
							deltas[indexLayer][indexWeight] = deltas[indexLayer][indexWeight] / float32(len(n.TrainingSet))
						}

					}
				}
			}


		}


		n.updateWeightsFromDeltas(deltas)
	}
	return errors
}



/*
TODO list for Thursday

TODO: Change the 'simple neural network' to be identical to the one in example of Peter to see if the values you get are correct
TODO: Deal with consequences of that - so that other tests end up correct
TODO: Create a numerical backpropagation function and compare gradients from both numerical and derivative version ( consider skipping if previous task works as it should )
TODO: Create a network that will test your training data in an 'offline way'
TODO: Create a way to observe how Total Cost is decreasing over time
TODO: Create a way to use the existing network on 'test' data and calculate accuracy. This should be another method in the network package




*/