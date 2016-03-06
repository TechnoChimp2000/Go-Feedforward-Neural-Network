package network


// interfaces
type Callback interface{
	ReceiveInfo(info string)
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

	trainer Trainer

}

func (n *NeuralNetwork)Train(callback Callback){
	n.trainer.train(callback, n)
}

func (n *NeuralNetwork)Calculate(trainingSampleInput []float32)[]float32{
	return n.feedForward(trainingSampleInput)
}




