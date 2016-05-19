package algebra

import (
	"testing"
	"fmt"
	"math"
)

func TestMatrixMultiply(t *testing.T){

	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
		[]float64{4, 6, 7},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	numbers2 := [][]float64{
		[]float64{8, 1, 2},
		[]float64{9, 5, 10},
		[]float64{11, -1, 12},
	}

	var matrix2 = &Matrix{Numbers:numbers2, NumOfRows:3, NumOfColumns:3}

	result := MultiplyMatrices(matrix1, matrix2)
	for i := 0; i < result.NumOfRows; i++ {
		fmt.Printf("Result: ", result.Numbers[i])
	}

}

func TestMatrixWithVectorMultiply(t *testing.T){

	numbers1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
		[]float64{10, 11, 12},
	}

	var matrix = &Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	vector := []float64{-2, 1, 0}

	result := Multiply(matrix, vector)

	fmt.Printf("Result: ", result)


}

func TestCreateVectorWithMeanAndStdDeviation(t *testing.T){
	normalizedVector := CreateVectorWithMeanAndStdDeviation(9,0, 1)

	var sum float64
	for _,value:=range normalizedVector{
		sum+=value
	}
	mean := sum/(float64)(len(normalizedVector))
	fmt.Printf("mean: ", mean)

	var sum2 float64
	for _,value:=range normalizedVector{
		sum2+=((value-mean)*(value-mean))
	}
	variance:=sum2/(float64)(len(normalizedVector))
	stDev := math.Sqrt((float64)(variance))
	fmt.Printf("stDev: ", stDev)

	fmt.Printf("Result: ", normalizedVector)
}

func TestNormalizedMatrix(t *testing.T){
	normalizedMatrix := CreateNormalizedMatrix(3, 4)
	fmt.Printf("Result: ", normalizedMatrix)
}

func simpleFunction(input float64)float64{
	return input * 2;
}

func TestVectorize(t *testing.T){
	vector := []float64{1,2,3}
	result := Vectorize(simpleFunction, vector)
	fmt.Printf("Result: ", result)
}

func TestAddVectors(t *testing.T){
	vector1 := []float64{1,2,3}
	vector2 := []float64{2,4,6}
	result := AddVectors(vector1, vector2)
	fmt.Printf("Result: ", result)

}

func TestCreateZerofiedMatrices(t *testing.T){

	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
		[]float64{4, 6, 7},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	numbers2 := [][]float64{
		[]float64{8, 1, 2},
		[]float64{9, 5, 10},
		[]float64{11, -1, 12},
	}

	var matrix2 = &Matrix{Numbers:numbers2, NumOfRows:3, NumOfColumns:3}

	matrices :=make([]*Matrix,2)
	matrices[0] = matrix1
	matrices[1] = matrix2
	result := CreateZerofiedMatrices(matrices)
	fmt.Printf("Result: ", result)

}

func TestCreateZerofiedDoubleArray(t *testing.T){
	numbers := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
		[]float64{4, 6, 7},
	}

	result := CreateZerofiedDoubleArray(numbers)
	fmt.Printf("Result: ", result)
}

func TestAddMatrices(t *testing.T){
	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:3, NumOfColumns:3}

	numbers2 := [][]float64{
		[]float64{8, 1, 2},
		[]float64{9, 5, 10},
		[]float64{11, -1, 12},
	}

	var matrix2 = &Matrix{Numbers:numbers2, NumOfRows:3, NumOfColumns:3}

	result := AddMatrices(matrix1, matrix2)
	fmt.Printf("Result: ", result)

}

func TestMultiplyMatrixWithNumber(t *testing.T){


	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:3, NumOfColumns:3}

	result := MultiplyMatrixWithNumber(matrix1, 2)

	fmt.Printf("Result: ", result)

}

func TestSubstractMatrices(t *testing.T){

	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:3, NumOfColumns:3}

	numbers2 := [][]float64{
		[]float64{8, 1, 2},
		[]float64{9, 5, 10},
		[]float64{11, -1, 12},
	}

	var matrix2 = &Matrix{Numbers:numbers2, NumOfRows:3, NumOfColumns:3}

	result := SubstractMatrices(matrix1, matrix2)
	fmt.Printf("Result: ", result)

}

func TestMultiplyVectorsWithNumber(t *testing.T){

	vector1 := []float64{1,2,3}
	result := MultiplyVectorsWithNumber(vector1, 2)
	fmt.Printf("Result: ", result)

}

func TestSubstractVectors(t *testing.T){

	vector1 := []float64{1,2,3}
	vector2 := []float64{2,4,6}
	result := SubstractVectors(vector1, vector2)
	fmt.Printf("Result: ", result)

}

func TestHadamard(t *testing.T){
	vector1 := []float64{1,2,3}
	vector2 := []float64{2,4,6}
	result := Hadamard(vector1, vector2)
	fmt.Printf("Result: ", result)
}

func TestCreateArrayWithDefaultValue(t *testing.T){
	result := CreateArrayWithDefaultValue(6,1)
	fmt.Printf("Result: ", result)

}

func TestMultiplyVectorWithTranspose(t *testing.T){

	vector1 := []float64{1,2,3}
	vector2 := []float64{2,4,6}
	result := MultiplyVectorWithTranspose(vector1, vector2)
	fmt.Printf("Result: ", result)

}

func TestTransposeMatrix(t *testing.T){

	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
		[]float64{4, 6, 7},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	result := TransposeMatrix(matrix1)
	fmt.Printf("Result: ", result)

}

func TestGetIndexOfMaxValue(t *testing.T){

	vector1 := []float64{1,2,3,4,3,2,1}
	result := GetIndexOfMaxValue(vector1)
	fmt.Printf("Result: ", result)

}

func TestReturnMatrixInSingleArray(t *testing.T){

	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
		[]float64{4, 6, 7},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	result := ReturnMatrixInSingleArray(matrix1)
	fmt.Printf("Result: ", result)

}

func TestReturnMatrixFromSingleArray(t *testing.T){
	numbers1 := [][]float64{
		[]float64{1, 2, 5},
		[]float64{-1, -2, -4},
		[]float64{0, 3, -3},
		[]float64{4, 6, 7},
	}

	var matrix1 = &Matrix{Numbers:numbers1, NumOfRows:4, NumOfColumns:3}

	array := ReturnMatrixInSingleArray(matrix1)

	result := ReturnMatrixFromSingleArray(4,3,array)

	fmt.Printf("Result: ", result)

}
