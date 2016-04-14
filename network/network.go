package network


// structures
type Connection struct{
	From          *Neuron
	To            *Neuron
	Weight        float32
}


type Neuron struct {
	// data
	output			float32
	InputConnections       	[]*Connection
	ConnectedToInNextLayer 	[]*Neuron
	// NEW STUFF
	Bias 			float32
	Delta 			float32
}

type NeuronLayer struct{
	Neurons []*Neuron
	layer   uint8
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

type Activation int

const (
	Logistic Activation = iota
	HyperbolicTangens
)

type NormalizerType int

const (
	Zscore NormalizerType = iota
	None
)

type CostFunctionType int

const (
	CrossEntrophy CostFunctionType = iota
	Quadratic
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

	trainer Trainer

	costFunction CostFunction

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


func (n *NeuralNetwork)SetCostFunction(costFunction CostFunctionType){
	switch costFunction {
	case CrossEntrophy:
		/**
		 * cross-entrophy cost function is only compatible with logistic activation function
		 */
		n.activationFunction = new(LogisticActivationFunction)
		n.costFunction = new(CrossEntrophyCostFunction)
	case Quadratic:
		n.costFunction = new(QuadraticCostFunction)
	}
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
		n.learningRate = 0.000001
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

func (n *NeuralNetwork)SetActivationFunction(activation Activation){
	//TODO: check if cost function is cross-entropy and switch to quadratic is activation is hyperbolic tangens - use reflect package

	switch activation {
	case Logistic:
		n.activationFunction = new(LogisticActivationFunction)
	case HyperbolicTangens:
		n.activationFunction = new(HyperbolicTangentActivationFunction)
	}
}

func (n *NeuralNetwork)GetActivationFunction() (function string) {
	// Igor TODO: This is a ghetto way of getting to know which Activation Function is being used. I used it because I could not figure out how to get that information in some other way

	if (n.activationFunction).min() == -1.0 {
		function = "HyperbolicTangensActivationFunction"
	} else {
		function = "LogisticActivationFunction"
	}

	return
}

func (n *NeuralNetwork)SetNormalizer(normalizer NormalizerType){
	switch normalizer {
	case Zscore:
		n.normalizer = new(ZscoreNormalizer)
	case None:
		n.normalizer = new(SkipNormalizer)
	}
}


func (n *NeuralNetwork)SetDebugMode(){
	n.debug = true
}




