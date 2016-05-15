package algebra
import "math"

func SigmoidDerivative(x []float32)[]float32{



	ones :=CreateArrayWithDefaultValue(len(x),1)
	sig := Sigmoid(x)
	substraction := SubstractVectors(ones, sig)
	return Hadamard(sig, substraction)

}



func Sigmoid (input []float32) []float32 {
	function :=  func(input float32) float32 {

		return float32(1.0/(1.0+math.Exp(-float64(input))))
	}
	return Vectorize(function, input)
}