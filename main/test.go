package main

import (
	"Go-Feedforward-Neural-Network/network"
	"fmt"

)

type CallbackReceiver struct{}
func (c CallbackReceiver) ReceiveInfo(info string){
	fmt.Println(info)

}

func main() {
	layer1 := network.CreateNetworkLayer(2, 0.35)
	layer2 := network.CreateNetworkLayer(2, 0.60)
	layer3 := network.CreateNetworkLayer(2, 0)

	for _,neuronInLayer1 := range layer1.Neurons{
		for _, neuronInLayer2 := range layer2.Neurons{
			neuronInLayer1.ConnectedToInNextLayer = append(neuronInLayer1.ConnectedToInNextLayer, neuronInLayer2)
		}
	}

	for _,neuronInLayer2 := range layer2.Neurons{
		for _, neuronInLayer3 := range layer3.Neurons{
			neuronInLayer2.ConnectedToInNextLayer = append(neuronInLayer2.ConnectedToInNextLayer, neuronInLayer3)
		}
	}

	w1 := &network.Connection{From: layer1.Neurons[0], To: layer2.Neurons[0], Weight: 0.15}
	layer2.Neurons[0].InputConnections = append(layer2.Neurons[0].InputConnections, w1)

	w2 := &network.Connection{From: layer1.Neurons[0], To: layer2.Neurons[1], Weight: 0.20}
	layer2.Neurons[1].InputConnections = append(layer2.Neurons[1].InputConnections, w2)

	w3 := &network.Connection{From: layer1.Neurons[1], To: layer2.Neurons[0], Weight: 0.25}
	layer2.Neurons[0].InputConnections = append(layer2.Neurons[0].InputConnections, w3)

	w4 := &network.Connection{From: layer1.Neurons[1], To: layer2.Neurons[1], Weight: 0.30}
	layer2.Neurons[1].InputConnections = append(layer2.Neurons[1].InputConnections, w4)

	w5 := &network.Connection{From: layer2.Neurons[0], To: layer3.Neurons[0], Weight: 0.40}
	layer3.Neurons[0].InputConnections = append(layer3.Neurons[0].InputConnections, w5)

	w6 := &network.Connection{From: layer2.Neurons[0], To: layer3.Neurons[1], Weight: 0.45}
	layer3.Neurons[1].InputConnections = append(layer3.Neurons[1].InputConnections, w6)

	w7 := &network.Connection{From: layer2.Neurons[1], To: layer3.Neurons[0], Weight: 0.50}
	layer3.Neurons[0].InputConnections = append(layer3.Neurons[0].InputConnections, w7)

	w8 := &network.Connection{From: layer2.Neurons[1], To: layer3.Neurons[1], Weight: 0.55}
	layer3.Neurons[1].InputConnections = append(layer3.Neurons[1].InputConnections, w8)

	var neuronLayers []*network.NeuronLayer
	neuronLayers = append(neuronLayers, & layer1)
	neuronLayers = append(neuronLayers, & layer2)
	neuronLayers = append(neuronLayers, & layer3)

	trainingInput := []float32{0.05, 0.10}
	trainingOutput := []float32{0.01, 0.99}

	trainingSample := network.TrainingSample{Input: trainingInput, Output: trainingOutput}

	trainingSamples := []network.TrainingSample{trainingSample}

	neuronNetwork := network.NeuralNetwork{NeuronLayers: neuronLayers, LearningRate: 0.5, TrainingSet: trainingSamples, Precision:0.05, ActivationFunction: new(network.LogisticActivationFunction)}

	neuronNetwork.TrainOnline(CallbackReceiver{})

}


