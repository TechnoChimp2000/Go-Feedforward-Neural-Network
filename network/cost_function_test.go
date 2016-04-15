package network

import (
	"testing"
	"fmt"
)
// Quadratic
func TestTotalErrorQuadratic (t *testing.T) {

	c := new(QuadraticCostFunction)

	actual := []float32{1.12, -0.123, 4.55, 85.1231231}
	output := []float32{0, 1, 0, 0}


	retval := c.calculateTotalError(actual,output)
	fmt.Println( "Testing total error for the quadratic function: ", retval) //908.64557

	if retval != 908.64557 {
		fmt.Println("Total error does not equal to 908.64557")
		t.Fail()
	}
}


func TestCalculateWeightDeltaInLastLayerQuadratic (t *testing.T) {

	c := new(QuadraticCostFunction)

	nn, trainingSamples := createSimpleNN() // 2,2,2

	// deltas before
	for _, neuron := range nn.neuronLayers[2].Neurons {
		//fmt.Printf("%v ", neuron.Delta)
		neuron.output = 2 // 2 was manually chosen. It's the value that outputs the following values as deltas: -3.98, -2.02
	}



	for neuronIndex, neuron := range nn.neuronLayers[2].Neurons {
		c.calculateWeightDeltaInLastLayer(nn, neuron, neuronIndex, 2, trainingSamples[0].Output )
	}

	// checks
	if ( nn.neuronLayers[2].Neurons[0].Delta != -3.98 ){
		fmt.Println("Failure at Layer: 2, Neuron: 0")
		fmt.Printf("Outputed delta: %v, Expected delta: %v\n", nn.neuronLayers[2].Neurons[0].Delta, -3.98 )
		t.Fail()
	}

	if ( nn.neuronLayers[2].Neurons[1].Delta != -2.02) {
		fmt.Println("Failure at Layer: 2, Neuron: 1")
		fmt.Printf("Outputed delta: %v, Expected delta: %v\n", nn.neuronLayers[2].Neurons[1].Delta, -2.02 )

		t.Fail()
	}
}

func TestCalculateWeightDeltaQuadratic (t *testing.T) {

	c := new(QuadraticCostFunction)
	nn, trainingSamples := createSimpleNN() // 2,2,2

	// setup neuron outputs manualy since we're not doing feed forward here

	nn.neuronLayers[1].Neurons[0].output = 2
	nn.neuronLayers[1].Neurons[1].output = 2

	nn.neuronLayers[2].Neurons[0].output = 2
	nn.neuronLayers[2].Neurons[1].output = 2

	nn.neuronLayers[2].Neurons[0].Delta = -3.98
	nn.neuronLayers[2].Neurons[1].Delta = -2.02

	/*
	fmt.Println("deltas before: ")
	for _, neuron := range nn.neuronLayers[1].Neurons {
		fmt.Printf("%v ", neuron.Delta)

	}
	*/

	for neuronIndex, neuron := range nn.neuronLayers[1].Neurons {
		c.calculateWeightDelta(nn, neuron, neuronIndex, 1, trainingSamples[0].Output )
	}

	/*
	fmt.Println("")
	fmt.Println("deltas after: ")
	for _, neuron := range nn.neuronLayers[1].Neurons {
		fmt.Printf("%v ", neuron.Delta)
	}
	*/

	if ( nn.neuronLayers[1].Neurons[0].Delta != -3.4258077 ){

		fmt.Println("Failure at Layer: 1, Neuron: 0")
		fmt.Printf("Outputed delta: %v, Expected delta: %v\n", nn.neuronLayers[1].Neurons[0].Delta, -3.4258077 )

		t.Fail()
	}

	if ( nn.neuronLayers[1].Neurons[1].Delta != -0.80281293 ) {

		fmt.Println("Failure at Layer: 1, Neuron: 1")
		fmt.Printf("Outputed delta: %v, Expected delta: %v\n", nn.neuronLayers[1].Neurons[1].Delta, -0.80281293 )

		t.Fail()
	}

}

// CrossEnthropy

/*
type CostFunction interface{
	calculateTotalError(actual []float32, output []float32) float32
	calculateWeightDeltaInLastLayer(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32)
	calculateWeightDelta(n *NeuralNetwork, neuron *Neuron, neuronIndex int, neuronLayerIndex int, trainingSampleOutput []float32)
}
*/

/*
func (q *QuadraticCostFunction) calculateTotalError(actual []float32, output []float32) float32 {
	var retval float32 = 0.0
	for outputIndex, singleOutput := range output{
		factor := singleOutput - actual[outputIndex]
		retval += 0.5 * factor * factor
	}
	return retval/float32(len(output))
}*/