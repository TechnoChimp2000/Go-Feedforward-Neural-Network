package cuda

import (
	"testing"
	"Go-Feedforward-Neural-Network/algebra"
	"fmt"
)


func TestMultiplyingMatrices(t *testing.T){
	MatricesMultiplication()
}

func TestMultiplyMatrices(t *testing.T){



	var matrix1 = algebra.CreateNormalizedMatrix(4,3)
	fmt.Printf("matrix1: ", matrix1)


	var matrix2 = algebra.CreateNormalizedMatrix(3,3)
	fmt.Printf("matrix2: ", matrix2)

	MultiplyMatrices(matrix1, matrix2)


}
