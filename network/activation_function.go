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
