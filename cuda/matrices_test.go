package cuda

import (
	"testing"
	"Go-Feedforward-Neural-Network/algebra"
	"fmt"
	"log"

	"time"
)


/*func TestMultiplyingMatrices(t *testing.T){
	MatricesMultiplication()
}*/

func TestMultiplyMatrices(t *testing.T){



	var matrix1 = algebra.CreateNormalizedMatrix(4,3)
	//fmt.Printf("matrix1: ", matrix1)


	var matrix2 = algebra.CreateNormalizedMatrix(3,3)
	//fmt.Printf("matrix2: ", matrix2)

	goResultMatrix := MultiplyMatrices(matrix1, matrix2)

	fmt.Printf("\n GPU result: ", goResultMatrix)

	cpuResult := algebra.MultiplyMatrices(matrix1, matrix2)
	fmt.Printf("\n CPU result: ",cpuResult)
	fmt.Println("")


}

func TestMultiplyBigMatricesOnGPU(t *testing.T){
	defer un(trace("TestMultiplyBigMatricesOnGPU"))

	var matrix1 = algebra.CreateNormalizedMatrix(1000,1000)
	var matrix2 = algebra.CreateNormalizedMatrix(1000,1000)
	MultiplyMatrices(matrix1, matrix2)

}

/*func TestMultiplyBigMatricesOnCPU(t *testing.T){
	defer un(trace("TestMultiplyBigMatricesOnCPU"))

	var matrix1 = algebra.CreateNormalizedMatrix(1000,1000)
	var matrix2 = algebra.CreateNormalizedMatrix(1000,1000)
	algebra.MultiplyMatrices(matrix1, matrix2)

}*/

func TestMultiplyDigitsMatricesOnGPU(t *testing.T){
	defer un(trace("TestMultiplyDigitsMatricesOnGPU"))

	var matrix1 = algebra.CreateNormalizedMatrix(30,784)
	var matrix2 = algebra.CreateNormalizedMatrix(784,10)
	MultiplyMatrices(matrix1, matrix2)

}

func TestMultiplyDigitsMatricesOnCPU(t *testing.T){
	defer un(trace("TestMultiplyDigitsMatricesOnCPU"))

	var matrix1 = algebra.CreateNormalizedMatrix(30,784)
	var matrix2 = algebra.CreateNormalizedMatrix(784,10)
	algebra.MultiplyMatrices(matrix1, matrix2)

}

func trace(s string) (string, time.Time) {
	log.Printf("trace start: %s\n", s)
	return s, time.Now()
}

func un(s string, startTime time.Time) {
	elapsed := time.Since(startTime)
	log.Printf("trace end: %s, elapsed %f secs\n", s, elapsed.Seconds())
}
