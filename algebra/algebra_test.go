package algebra

import (
	"testing"
	"fmt"
)

func TestMatrixMultiply(t *testing.T){

	numbers1 := [][]float32{
		[]float32{1, 2, 5},
		[]float32{-1, -2, -4},
		[]float32{0, 3, -3},
		[]float32{4, 6, 7},
	}

	var matrix1 = &Matrix{numbers:numbers1, numOfRows:4, numOfColumns:3}

	numbers2 := [][]float32{
		[]float32{8, 1, 2},
		[]float32{9, 5, 10},
		[]float32{11, -1, 12},
	}

	var matrix2 = &Matrix{numbers:numbers2, numOfRows:3, numOfColumns:3}

	result := MultiplyMatrices(matrix1, matrix2)
	for i := 0; i < result.numOfRows; i++ {
		fmt.Printf("Result: ", result.numbers[i])
	}

}

func TestMatrixWithVectorMultiply(t *testing.T){

	numbers1 := [][]float32{
		[]float32{1, 2, 3},
		[]float32{4, 5, 6},
		[]float32{7, 8, 9},
		[]float32{10, 11, 12},
	}

	var matrix = &Matrix{numbers:numbers1, numOfRows:4, numOfColumns:3}

	vector := []float32{-2, 1, 0}

	result := Multiply(matrix, vector)

	fmt.Printf("Result: ", result)


}

func TestNormalizedVector(t *testing.T){
	normalizedVector := CreateNormalizedVector(10)
	fmt.Printf("Result: ", normalizedVector)
}

func TestNormalizedMatrix(t *testing.T){
	normalizedMatrix := CreateNormalizedMatrix(3, 4)
	fmt.Printf("Result: ", normalizedMatrix)
}

func simpleFunction(input float32)float32{
	return input * 2;
}

func TestVectorize(t *testing.T){
	vector := []float32{1,2,3}
	result := Vectorize(simpleFunction, vector)
	fmt.Printf("Result: ", result)
}

func TestAddVectors(t *testing.T){
	vector1 := []float32{1,2,3}
	vector2 := []float32{2,4,6}
	result := AddVectors(vector1, vector2)
	fmt.Printf("Result: ", result)

}
