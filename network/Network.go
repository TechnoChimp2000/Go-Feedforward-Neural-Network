package network


// interfaces
type PropagationFunction interface{
	Propagate(_weights []float32, _signals []float32) float32
}

type ActivationFunction interface{
	Activate (_input float32) float32
}

type ConnectionInitializer interface{
	Initialize(from Neuron, to Neuron) float32
}

type RandomConnectionInitializer ConnectionInitializer

func(r RandomConnectionInitializer) Initialize(from Neuron, to Neuron) float32{
	//TODO write random function that returns value between -1 and 1
	return 0.0
}


// structures
type Connection struct{
	from Neuron
	to Neuron
	weight float32
	connectionInitializer ConnectionInitializer
}

func (c Connection) InitializeWeight(){
	c.weight = c.connectionInitializer.Initialize(c.from, c.to)
}

type Neuron struct {
	// data
	output float32
	// functions
	activationFunction ActivationFunction
	propagationFunction PropagationFunction
}

type NeuronLayer struct{
	neurons []Neuron
	layer uint8
}


type TrainingSample struct{
	input []float32
	output []float32
}

type ConnectionLayer struct{
	connections []Connection
	layer uint8

}

type NeuralNetwork struct{
	neuronLayers []NeuronLayer
	learningRate float32
	trainingSet []TrainingSample
	connectionLayers []ConnectionLayer
}

func (n NeuralNetwork) InitializeConnections(){
	for _,connectionLayer := range n.connectionLayers {
		for _,connection := range connectionLayer.connections {
			connection.InitializeWeight()
		}
	}
}

func (n NeuralNetwork) TrainOnline(){
	n.InitializeConnections()
}

func (n NeuralNetwork) TrainOffline(){
	n.InitializeConnections()
}
