package network

func CreateNetworkLayer(neuronNumber int, bias float32) NeuronLayer{
	var neurons []Neuron
	for i :=0; i< neuronNumber; i++ {
		neurons = append(neurons, Neuron{})
	}
	return NeuronLayer{Neurons: neurons, Bias: bias}
}
