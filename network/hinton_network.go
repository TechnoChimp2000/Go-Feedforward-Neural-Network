package network
/*
type HintonNetwork struct {
	layers []int

	//hintonWeights should be floats and otherwise nils
	hintonWeights [][]float64
	activationFunction ActivationFunction
	propagationFunction PropagationFunction

	//training set should also have bias 1s - for now
	trainingSet []TrainingSample
}

//For now hintonWeights are considered to have bias neuron on each layer
//TODO think if biases should be added in Train method
func (h *HintonNetwork) Train() [][]float64{
	//TODO: input validation



	return h.hintonWeights
}

func (h *HintonNetwork) feedForward(input []float32) []float64{
	getWeight := func(hintonIndex int, weightIndex int) float64{
		weights := h.hintonWeights[hintonIndex]
		count := 0
		for z:=0; z<len(weights); z++{
			if(weights[z]!=nil){
				if count == weightIndex {
					return weights[z]
				}
				count++;
			}
		}
		panic("No weight found for hinton index: "+hintonIndex+" and weight index: "+weightIndex)

	}

	calculateOffset := func(i int) int {
		offset := 0
		for j:=0; j<i; j++{
			offset+=h.layers[j]
		}
		return offset
	}

	for i := 1; i < len(h.layers); i++ {
		for hintonIndex:=h.layers[i-1] + calculateOffset(i-1); hintonIndex < h.layers[i]+calculateOffset(i); hintonIndex++ {
			net := 0
			for inputIndex, inputSingle := range input{
				net += inputSingle * getWeight(hintonIndex, inputIndex)
			}
		}
	}



	return h.hintonWeights
}
*/
/*func (h *HintonNetwork) addBiases() [][]float64{
	//TODO: input validation

	for i := 1; i < len(h.layers); i++ {
		for j:=0; j < h.layers[i]; j++ {

		}
	}



	return h.hintonWeights
}*/


