package network

func CreateNeuronLayer(neuronNumber int, NumberOfInputConnections int, bias float32) NeuronLayer{
	var neurons []*Neuron
	for i :=0; i< neuronNumber; i++ {
		neurons = append(neurons, new(Neuron))
	}
	return NeuronLayer{Neurons: neurons, Bias: bias}
}
