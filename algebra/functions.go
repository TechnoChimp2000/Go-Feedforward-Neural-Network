package algebra
import "math"

func SigmoidDerivative(x []float64)[]float64{



	ones :=CreateArrayWithDefaultValue(len(x),1)
	sig := Sigmoid(x)
	substraction := SubstractVectors(ones, sig)
	return Hadamard(sig, substraction)

}



func Sigmoid (input []float64) []float64 {
	function :=  func(input float64) float64 {

		return float64(1.0/(1.0+math.Exp(-float64(input))))
	}
	return Vectorize(function, input)
}