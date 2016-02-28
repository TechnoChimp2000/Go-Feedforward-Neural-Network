package network

import (
	"testing"
	"fmt"
	"strconv"
)

type CallbackReceiver struct{}
func (c CallbackReceiver) ReceiveInfo(info string){
	//fmt.Println(info)

}

func TestSimpleNN(t *testing.T){
	layer1 := CreateNeuronLayer(2, 2,  0.35)
	layer2 := CreateNeuronLayer(2, 4, 0.60)
	layer3 := CreateNeuronLayer(2, 4, 0)

	for _,neuronInLayer1 := range layer1.Neurons {
		for _, neuronInLayer2 := range layer2.Neurons {
			neuronInLayer1.ConnectedToInNextLayer = append(neuronInLayer1.ConnectedToInNextLayer, neuronInLayer2)
		}
	}

	for _,neuronInLayer2 := range layer2.Neurons {
		for _, neuronInLayer3 := range layer3.Neurons {
			neuronInLayer2.ConnectedToInNextLayer = append(neuronInLayer2.ConnectedToInNextLayer, neuronInLayer3)
		}
	}

	w1 := &Connection{From: layer1.Neurons[0], To: layer2.Neurons[0], Weight: 0.15}
	layer2.Neurons[0].InputConnections = append(layer2.Neurons[0].InputConnections, w1)

	w2 := &Connection{From: layer1.Neurons[0], To: layer2.Neurons[1], Weight: 0.20}
	layer2.Neurons[1].InputConnections = append(layer2.Neurons[1].InputConnections, w2)

	w3 := &Connection{From: layer1.Neurons[1], To: layer2.Neurons[0], Weight: 0.25}
	layer2.Neurons[0].InputConnections = append(layer2.Neurons[0].InputConnections, w3)

	w4 := &Connection{From: layer1.Neurons[1], To: layer2.Neurons[1], Weight: 0.30}
	layer2.Neurons[1].InputConnections = append(layer2.Neurons[1].InputConnections, w4)

	w5 := &Connection{From: layer2.Neurons[0], To: layer3.Neurons[0], Weight: 0.40}
	layer3.Neurons[0].InputConnections = append(layer3.Neurons[0].InputConnections, w5)

	w6 := &Connection{From: layer2.Neurons[0], To: layer3.Neurons[1], Weight: 0.45}
	layer3.Neurons[1].InputConnections = append(layer3.Neurons[1].InputConnections, w6)

	w7 := &Connection{From: layer2.Neurons[1], To: layer3.Neurons[0], Weight: 0.50}
	layer3.Neurons[0].InputConnections = append(layer3.Neurons[0].InputConnections, w7)

	w8 := &Connection{From: layer2.Neurons[1], To: layer3.Neurons[1], Weight: 0.55}
	layer3.Neurons[1].InputConnections = append(layer3.Neurons[1].InputConnections, w8)

	var neuronLayers []*NeuronLayer
	neuronLayers = append(neuronLayers, & layer1)
	neuronLayers = append(neuronLayers, & layer2)
	neuronLayers = append(neuronLayers, & layer3)

	trainingInput := []float32{0.05, 0.10}
	trainingOutput := []float32{0.01, 0.99}
	//trainingOutput := []float32{0.5, 0.5}

	trainingSample := TrainingSample{Input: trainingInput, Output: trainingOutput}

	trainingSamples := []TrainingSample{trainingSample}

	neuronNetwork := NeuralNetwork{NeuronLayers: neuronLayers, LearningRate: 0.02, TrainingSet: trainingSamples, Precision:0.0003, ActivationFunction: new(LogisticActivationFunction)}

	neuronNetwork.TrainOnline(CallbackReceiver{})

	result := neuronNetwork.FeedForward(trainingInput)
	fmt.Println("network test result: "+strconv.FormatFloat(float64(result[0]), 'E', -1, 32) + " " + strconv.FormatFloat(float64(result[1]), 'E', -1, 32))







}
