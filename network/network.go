package network


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

type Precision int

const (
	Rough Precision = iota
	Medium
	High
	VeryHigh

)

type LearningRate int

const (
	Fast LearningRate = iota
	Normal
	Slow
	VerySlow

)

type TrainerMode int

const (
	Online TrainerMode = iota
	Offline
)


type NeuralNetwork struct{
	neuronLayers       []*NeuronLayer
	learningRate       float32

	//precision tells how precise network should  be
	//that is error should be lesser than precision
	//typical value is 0.05
	precision          float32


	// functions
	activationFunction ActivationFunction

	normalizer Normalizer

	trainer            Trainer

	//if debug is true nn processing is logged
	debug              bool

}

func (n *NeuralNetwork)Train(trainingSet        []TrainingSample){
	normalizeTrainingInput(trainingSet, n.normalizer)
	n.trainer.train(n, trainingSet)
}

func (n *NeuralNetwork)Calculate(input []float32)[]float32{
	normalizedInput	:= n.normalizer.normalizeVector(input)
	return n.feedForward(normalizedInput)
}

func (n *NeuralNetwork)SetPrecision(precision Precision){
	switch precision {
	case Rough:
		n.precision = 0.01
	case Medium:
		n.precision = 0.005
	case High:
		n.precision = 0.001
	case VeryHigh:
		n.precision = 0.0001
	}
}

func (n *NeuralNetwork)SetLearningRate(learningRate LearningRate){
	switch learningRate {
	case Fast:
		n.learningRate = 0.01
	case Normal:
		n.learningRate = 0.02
	case Slow:
		n.learningRate = 0.005
	case VerySlow:
		n.learningRate = 0.001
	}
}

func (n *NeuralNetwork)SetTrainerMode(trainerMode TrainerMode){
	switch trainerMode {
	case Online:
		n.trainer = new(OnlineTrainer)
	case Offline:
		n.trainer = new(OfflineTrainer)
	}
}


func (n *NeuralNetwork)SetDebugMode(){
	n.debug = true
}




