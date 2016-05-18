package network2

import "Go-Feedforward-Neural-Network/algebra"

type CostFunction interface{
	calculateDelta(activation ,output, weightedInput []float64) []float64
}

type QuadraticCostFunction struct{}

func(q *QuadraticCostFunction)calculateDelta(activation ,output, weightedInput []float64) []float64{
	return algebra.Hadamard(costDerivative(activation, output), algebra.SigmoidDerivative(weightedInput))
}

type CrossEntrophyCostFunction struct{}

func(c *CrossEntrophyCostFunction)calculateDelta(activation ,output, weightedInput []float64) []float64{
	return costDerivative(activation, output)
}


