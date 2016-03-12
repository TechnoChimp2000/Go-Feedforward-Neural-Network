package network
import "math"

type ActivationFunction interface{
	Activate (input float32) float32
	Derivative(input float32) float32
	min() float32
	max() float32
}

type LogisticActivationFunction struct{}

func (l *LogisticActivationFunction) Activate (input float32) float32 {
	return float32(1.0/(1.0+math.Exp(-float64(input))))
}

func (l *LogisticActivationFunction) Derivative (input float32) float32 {
	return input * (1 - input)
}

func (l *LogisticActivationFunction) min() float32 {
	return 0.0
}

func (l *LogisticActivationFunction) max() float32 {
	return 1.0
}

type HyperbolicTangentActivationFunction struct {}

func (h *HyperbolicTangentActivationFunction) Activate (input float32) float32 {
	return float32(1.7159 * math.Atan(float64((2/float32(3)) * input)))
}

func (h *HyperbolicTangentActivationFunction) Derivative (input float32) float32 {
	return float32(51477/float32((20000*input*input) + 45000))
}

func (h *HyperbolicTangentActivationFunction) min() float32 {
	return -1.0
}

func (h *HyperbolicTangentActivationFunction) max() float32 {
	return 1.0
}
