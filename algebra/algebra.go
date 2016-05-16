package algebra

import (

	"math/rand"
	"fmt"
	"time"
"math"
)

type Matrix struct{
	numbers [][]float32
	numOfColumns int
	numOfRows int
}

func (matrix *Matrix) String() string {
	return fmt.Sprintf("", matrix.numbers)
}

func Multiply(matrix *Matrix, vector []float32)[]float32{
	matrix2 := createMatrixFromVector(vector)
	product := MultiplyMatrices(matrix, matrix2)
	return createVectorFromMatrix(product)
}

func createVectorFromMatrix(matrix *Matrix)[]float32{
	vector :=make([]float32, matrix.numOfRows)
	for i:=0; i<matrix.numOfRows;i++{
		vector[i] = matrix.numbers[i][0]
	}
	return vector
}

func createMatrixFromVector(vector []float32)*Matrix{
	numbers := make([][]float32, len(vector))
	for i := 0; i < len(vector); i++ {
		numbers[i] = make([]float32, 1)
		numbers[i][0] = vector[i]
	}
	matrix :=&Matrix{numOfRows:len(vector), numOfColumns: 1, numbers: numbers}
	return matrix
}

func MultiplyMatrices(matrix1, matrix2 *Matrix) *Matrix{
	numbers := make([][]float32, matrix1.numOfRows)
	for i := 0; i < matrix1.numOfRows; i++ {
		numbers[i] = make([]float32, matrix2.numOfColumns)
	}
	var result = &Matrix{numOfRows: matrix1.numOfRows, numOfColumns: matrix2.numOfColumns, numbers: numbers}
	for i:=0; i<matrix1.numOfRows; i++{
		for j:=0; j<matrix2.numOfColumns; j++{
			column := getMatrixColumn(j, matrix2)
			result.numbers[i][j] = calculateMatrixNumber(matrix1.numbers[i], column)//
		}
	}

	return result

}

func calculateMatrixNumber(row, column []float32)float32{
	var result  float32
	for i:=0; i<len(row); i++ {
		result+=(row[i]*column[i])
	}
	return result
}

func getMatrixColumn(colNum int, matrix *Matrix)[]float32{
	size := matrix.numOfRows
	var result = make([]float32,size)
	for i:=0; i<size; i++ {
		result[i] = matrix.numbers[i][colNum]
	}

	return result
}

/*func CreateNormalizedMatrix(numOfRows int, numOfColumns int) *Matrix{
	normalizedVector := CreateNormalizedVector(numOfColumns * numOfRows)
	z := 0
	numbers := make([][]float32, numOfRows)
	for i := 0; i < numOfRows; i++ {
		numbers[i] = make([]float32, numOfColumns)
		for j:=0; j<numOfColumns; j++{
			numbers[i][j] = normalizedVector[z]
			z++
		}


	}
	var result = &Matrix{numOfRows: numOfRows, numOfColumns: numOfColumns, numbers: numbers}
	return result

}*/

func CreateNormalizedMatrix(numOfRows int, numOfColumns int) *Matrix{

	numbers := make([][]float32, numOfRows)
	for i := 0; i < numOfRows; i++ {
		numbers[i] = CreateVectorWithMeanAndStdDeviation(numOfColumns, 0,1.0/(float32)(math.Sqrt((float64)(numOfColumns))))
	}
	var result = &Matrix{numOfRows: numOfRows, numOfColumns: numOfColumns, numbers: numbers}
	return result

}


func CreateVectorWithMeanAndStdDeviation(length int, mean float32, stdDeviation float32) []float32 {

	vector := make([]float32, length)
	/**
	 * randomize vector first
	 */
	rand.Seed(time.Now().UTC().UnixNano())
	for i:=0; i<length; i++{
		vector[i] = (float32)(rand.NormFloat64())
	}


	result := make([]float32, len(vector))
	oldMean := calculateMean(vector)

	oldStdDeviation := calculateStdDeviation(vector, oldMean)

	for i,_ :=range vector{
		result[i] = (mean + ((vector[i] - oldMean)*(stdDeviation / oldStdDeviation)))
	}
	return result
}

func calculateMean(vector []float32)float32{
	var sum float32
	for _,value:=range vector{
		sum+=value
	}
	mean := sum/(float32)(len(vector))
	return mean
}

func calculateStdDeviation(vector []float32, mean float32)float32{
	var sum2 float32
	for _,value:=range vector{
		sum2+=((value-mean)*(value-mean))
	}
	variance:=sum2/(float32)(len(vector))
	stDev := (float32)(math.Sqrt((float64)(variance)))
	return stDev
}

func Vectorize(function func(float32)float32, vector []float32) []float32{
	result := make([]float32, len(vector))

	for i, value := range vector{
		result[i] = function(value)
	}
	return result
}

func MultiplyVectorArrayWithNumber(vectorArray [][]float32, number float32)[][]float32{

	result := make([][]float32, len(vectorArray))
	for i := 0; i < len(vectorArray); i++ {
		result[i] = MultiplyVectorsWithNumber(vectorArray [i], number)
	}
	return result
}


func MultiplyVectorsWithNumber(vector []float32, number float32)[]float32{
	result := make([]float32, len(vector))

	for i, value := range vector{
		result[i] = (number * value)
	}
	return result
}


func AddVectorArrays(vectorArray1, vectorArray2 [][]float32)[][]float32{
	result := make([][]float32, len(vectorArray1))

	for i, _ := range vectorArray1{

		result[i] = AddVectors(vectorArray1[i], vectorArray2[i])
	}
	return result

}

func Hadamard(vector1, vector2 []float32)[]float32{
	result := make([]float32, len(vector1))

	for i, value := range vector1{
		result[i] = value * vector2[i]
	}
	return result

}

func AddVectors(vector1, vector2 []float32)[]float32{
	result := make([]float32, len(vector1))

	for i, value := range vector1{
		result[i] = value + vector2[i]
	}
	return result

}

func SubstractVectorArrays(vectorArray1, vectorArray2 [][]float32)[][]float32{
	result := make([][]float32, len(vectorArray1))

	for i, _ := range vectorArray1{

		result[i] = SubstractVectors(vectorArray1[i], vectorArray2[i])
	}
	return result

}

func SubstractVectors(vector1, vector2 []float32)[]float32{

	return AddVectors(vector1, MultiplyVectorsWithNumber(vector2, -1))

}

func AddMatriceArrays(matrixArray1, matrixArray2 []*Matrix)[]*Matrix{
	result := make([]*Matrix, len(matrixArray1))

	for i, _ := range matrixArray1{

		result[i] = AddMatrices(matrixArray1[i], matrixArray2[i])
	}
	return result

}

func AddMatrices(matrix1, matrix2 *Matrix)*Matrix{
	numbers := make([][]float32, matrix1.numOfRows)
	for i := 0; i < matrix1.numOfRows; i++ {
		numbers[i] = make([]float32, matrix1.numOfColumns)
	}
	var result = &Matrix{numOfRows: matrix1.numOfRows, numOfColumns: matrix1.numOfColumns, numbers: numbers}
	for i:=0; i<matrix1.numOfRows; i++{
		for j:=0; j<matrix1.numOfColumns; j++{

			result.numbers[i][j] = matrix1.numbers[i][j] + matrix2.numbers[i][j]
		}
	}

	return result
}

func SubstractMatriceArrays(matrixArray1, matrixArray2 []*Matrix)[]*Matrix{
	result := make([]*Matrix, len(matrixArray1))

	for i, _ := range matrixArray1{

		result[i] = SubstractMatrices(matrixArray1[i], matrixArray2[i])
	}
	return result

}

func SubstractMatrices(matrix1, matrix2 *Matrix)*Matrix{
	return AddMatrices(matrix1, MultiplyMatrixWithNumber(matrix2, -1))
}

func MultiplyMatrixArrayWithNumber(matrixArray []*Matrix, number float32)[]*Matrix{

	result := make([]*Matrix, len(matrixArray))

	for i, _ := range matrixArray{

		result[i] = MultiplyMatrixWithNumber(matrixArray[i], number)
	}
	return result

}

func MultiplyMatrixWithNumber(matrix *Matrix, number float32)*Matrix{
	numbers := make([][]float32, matrix.numOfRows)
	for i := 0; i < matrix.numOfRows; i++ {
		numbers[i] = make([]float32, matrix.numOfColumns)
	}
	var result = &Matrix{numOfRows: matrix.numOfRows, numOfColumns: matrix.numOfColumns, numbers: numbers}

	for i:=0; i<matrix.numOfRows; i++{
		for j:=0; j<matrix.numOfColumns; j++{

			result.numbers[i][j] = (number * matrix.numbers[i][j])
		}
	}

	return result

}

func CreateZerofiedMatrices(values []*Matrix)[]*Matrix{
	result := make([]*Matrix, len(values))

	for j:=0; j<len(values); j++{
		numbers := make([][]float32, values[j].numOfRows)
		for i := 0; i < values[j].numOfRows; i++ {
			numbers[i] = make([]float32, values[j].numOfColumns)
		}
		result[j] = &Matrix{numOfRows: values[j].numOfRows, numOfColumns: values[j].numOfColumns, numbers: numbers}
	}

	return result
}

func CreateZerofiedDoubleArray(values [][]float32)[][]float32{
	result := make([][]float32, len(values))
	for i := 0; i < len(values); i++ {
		result[i] = make([]float32, len(values[i]))
	}
	return result
}

func CreateArrayWithDefaultValue(length int, value float32)[]float32{
	result := make([]float32, length)
	for i := 0; i < length; i++ {
		result[i] = value
	}
	return result
}

func MultiplyVectorWithTranspose(vector1, vector2 []float32)*Matrix{

	createMatrixFromVector(vector1)

	return MultiplyMatrices(createMatrixFromVector(vector1), createTransposeMatrixFromVector(vector2))



}
func createTransposeMatrixFromVector(vector []float32)*Matrix{
	numbers := make([][]float32, 1)
	numbers[0] = make([]float32, len(vector))
	for i := 0; i < len(vector); i++ {
		numbers[0][i] = vector[i]
	}
	matrix :=&Matrix{numOfRows:1, numOfColumns: len(vector), numbers: numbers}
	return matrix
}

func TransposeMatrix(matrix *Matrix)*Matrix{
	numbers := make([][]float32, matrix.numOfColumns)
	for i := 0; i < matrix.numOfColumns; i++ {
		numbers[i] = make([]float32, matrix.numOfRows)
		for j:=0; j<matrix.numOfRows;j++{
			numbers[i][j] = matrix.numbers[j][i]
		}

	}
	var result = &Matrix{numOfRows: matrix.numOfColumns, numOfColumns: matrix.numOfRows, numbers: numbers}

	return result
}

func GetIndexOfMaxValue(vector []float32)int{
	index :=0
	max := vector[0]
	for i,value := range(vector){
		if(value>max){
			index = i
			max = value
		}

	}
	return index
}
