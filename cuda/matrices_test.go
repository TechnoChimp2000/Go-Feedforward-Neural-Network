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

func TestMultiplyMatrixWithVector(t *testing.T){
	numbers1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
		[]float64{10, 11, 12},
	}

	var matrix = &algebra.Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	vector := []float32{-2, 1, 0}

	gpuResult := MultiplyMatrixWithVector(matrix, vector)

	fmt.Printf("\n GPU result matrix with vector is: ", gpuResult)

	cpuResult := algebra.Multiply(matrix, convert32to64(vector))

	fmt.Printf("\n CPU result matrix with vector is: ", cpuResult)

}

func TestMultiplyBigMatrixWithVectorOnGPU(t *testing.T){
	defer un(trace("TestMultiplyBigMatrixWithVectorOnGPU"))

	var matrix = algebra.CreateNormalizedMatrix(1000,1000)
	var vector = algebra.CreateVectorWithMeanAndStdDeviation(1000,0,1)
	MultiplyMatrixWithVector(matrix, convert64to32(vector))

}

func TestMultiplyBigMatrixWithVectorOnCPU(t *testing.T){
	defer un(trace("TestMultiplyBigMatrixWithVectorOnCPU"))

	var matrix = algebra.CreateNormalizedMatrix(1000,1000)
	var vector = algebra.CreateVectorWithMeanAndStdDeviation(10000,0,1)
	algebra.Multiply(matrix, vector)

}

func convert64to32(input []float64)[]float32{
	result := make([]float32, len(input))
	for i,_ := range input{
		result[i] = (float32)(input[i])
	}
	return result
}

func convert32to64(input []float32)[]float64{
	result := make([]float64, len(input))
	for i,_ := range input{
		result[i] = (float64)(input[i])
	}
	return result
}

func trace(s string) (string, time.Time) {
	log.Printf("trace start: %s\n", s)
	return s, time.Now()
}

func un(s string, startTime time.Time) {
	elapsed := time.Since(startTime)
	log.Printf("trace end: %s, elapsed %f secs\n", s, elapsed.Seconds())
}
