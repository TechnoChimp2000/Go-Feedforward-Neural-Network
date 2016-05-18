package algebra

import (

	"math/rand"
	"fmt"
	"time"
"math"
)

type Matrix struct{
	Numbers      [][]float64
	NumOfColumns int
	NumOfRows    int
}

func (matrix *Matrix) String() string {
	return fmt.Sprintf("", matrix.Numbers)
}

func ReturnMatrixInSingleArray(matrix *Matrix)[]float32{
	result := make([]float32, matrix.NumOfRows * matrix.NumOfColumns)

	for i:=0; i<matrix.NumOfRows;i++{
		for j:=0; j<matrix.NumOfColumns;j++{
			result[i+ (j*matrix.NumOfRows)] = (float32)(matrix.Numbers[i][j])
		}

	}
	return result
}

func Multiply(matrix *Matrix, vector []float64)[]float64{
	matrix2 := createMatrixFromVector(vector)
	product := MultiplyMatrices(matrix, matrix2)
	return createVectorFromMatrix(product)
}

func createVectorFromMatrix(matrix *Matrix)[]float64{
	vector :=make([]float64, matrix.NumOfRows)
	for i:=0; i<matrix.NumOfRows;i++{
		vector[i] = matrix.Numbers[i][0]
	}
	return vector
}

func createMatrixFromVector(vector []float64)*Matrix{
	numbers := make([][]float64, len(vector))
	for i := 0; i < len(vector); i++ {
		numbers[i] = make([]float64, 1)
		numbers[i][0] = vector[i]
	}
	matrix :=&Matrix{NumOfRows:len(vector), NumOfColumns: 1, Numbers: numbers}
	return matrix
}

func MultiplyMatrices(matrix1, matrix2 *Matrix) *Matrix{
	numbers := make([][]float64, matrix1.NumOfRows)
	for i := 0; i < matrix1.NumOfRows; i++ {
		numbers[i] = make([]float64, matrix2.NumOfColumns)
	}
	var result = &Matrix{NumOfRows: matrix1.NumOfRows, NumOfColumns: matrix2.NumOfColumns, Numbers: numbers}
	for i:=0; i<matrix1.NumOfRows; i++{
		for j:=0; j<matrix2.NumOfColumns; j++{
			column := getMatrixColumn(j, matrix2)
			result.Numbers[i][j] = calculateMatrixNumber(matrix1.Numbers[i], column)//
		}
	}

	return result

}

func calculateMatrixNumber(row, column []float64)float64{
	var result  float64
	for i:=0; i<len(row); i++ {
		result+=(row[i]*column[i])
	}
	return result
}

func getMatrixColumn(colNum int, matrix *Matrix)[]float64{
	size := matrix.NumOfRows
	var result = make([]float64,size)
	for i:=0; i<size; i++ {
		result[i] = matrix.Numbers[i][colNum]
	}

	return result
}

/*func CreateNormalizedMatrix(numOfRows int, numOfColumns int) *Matrix{
	normalizedVector := CreateNormalizedVector(numOfColumns * numOfRows)
	z := 0
	numbers := make([][]float64, numOfRows)
	for i := 0; i < numOfRows; i++ {
		numbers[i] = make([]float64, numOfColumns)
		for j:=0; j<numOfColumns; j++{
			numbers[i][j] = normalizedVector[z]
			z++
		}


	}
	var result = &Matrix{numOfRows: numOfRows, numOfColumns: numOfColumns, numbers: numbers}
	return result

}*/

func CreateNormalizedMatrix(numOfRows int, numOfColumns int) *Matrix{

	numbers := make([][]float64, numOfRows)
	for i := 0; i < numOfRows; i++ {
		numbers[i] = CreateVectorWithMeanAndStdDeviation(numOfColumns, 0,1.0/(float64)(math.Sqrt((float64)(numOfColumns))))
	}
	var result = &Matrix{NumOfRows: numOfRows, NumOfColumns: numOfColumns, Numbers: numbers}
	return result

}


func CreateVectorWithMeanAndStdDeviation(length int, mean float64, stdDeviation float64) []float64 {

	vector := make([]float64, length)
	/**
	 * randomize vector first
	 */
	rand.Seed(time.Now().UTC().UnixNano())
	for i:=0; i<length; i++{
		vector[i] = (float64)(rand.NormFloat64())
	}


	result := make([]float64, len(vector))
	oldMean := calculateMean(vector)

	oldStdDeviation := calculateStdDeviation(vector, oldMean)

	for i,_ :=range vector{
		result[i] = (mean + ((vector[i] - oldMean)*(stdDeviation / oldStdDeviation)))
	}
	return result
}

func calculateMean(vector []float64)float64{
	var sum float64
	for _,value:=range vector{
		sum+=value
	}
	mean := sum/(float64)(len(vector))
	return mean
}

func calculateStdDeviation(vector []float64, mean float64)float64{
	var sum2 float64
	for _,value:=range vector{
		sum2+=((value-mean)*(value-mean))
	}
	variance:=sum2/(float64)(len(vector))
	stDev := (float64)(math.Sqrt((float64)(variance)))
	return stDev
}

func Vectorize(function func(float64)float64, vector []float64) []float64{
	result := make([]float64, len(vector))

	for i, value := range vector{
		result[i] = function(value)
	}
	return result
}

func MultiplyVectorArrayWithNumber(vectorArray [][]float64, number float64)[][]float64{

	result := make([][]float64, len(vectorArray))
	for i := 0; i < len(vectorArray); i++ {
		result[i] = MultiplyVectorsWithNumber(vectorArray [i], number)
	}
	return result
}


func MultiplyVectorsWithNumber(vector []float64, number float64)[]float64{
	result := make([]float64, len(vector))

	for i, value := range vector{
		result[i] = (number * value)
	}
	return result
}


func AddVectorArrays(vectorArray1, vectorArray2 [][]float64)[][]float64{
	result := make([][]float64, len(vectorArray1))

	for i, _ := range vectorArray1{

		result[i] = AddVectors(vectorArray1[i], vectorArray2[i])
	}
	return result

}

func Hadamard(vector1, vector2 []float64)[]float64{
	result := make([]float64, len(vector1))

	for i, value := range vector1{
		result[i] = value * vector2[i]
	}
	return result

}

func AddVectors(vector1, vector2 []float64)[]float64{
	result := make([]float64, len(vector1))

	for i, value := range vector1{
		result[i] = value + vector2[i]
	}
	return result

}

func SubstractVectorArrays(vectorArray1, vectorArray2 [][]float64)[][]float64{
	result := make([][]float64, len(vectorArray1))

	for i, _ := range vectorArray1{

		result[i] = SubstractVectors(vectorArray1[i], vectorArray2[i])
	}
	return result

}

func SubstractVectors(vector1, vector2 []float64)[]float64{

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
	numbers := make([][]float64, matrix1.NumOfRows)
	for i := 0; i < matrix1.NumOfRows; i++ {
		numbers[i] = make([]float64, matrix1.NumOfColumns)
	}
	var result = &Matrix{NumOfRows: matrix1.NumOfRows, NumOfColumns: matrix1.NumOfColumns, Numbers: numbers}
	for i:=0; i<matrix1.NumOfRows; i++{
		for j:=0; j<matrix1.NumOfColumns; j++{

			result.Numbers[i][j] = matrix1.Numbers[i][j] + matrix2.Numbers[i][j]
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

func MultiplyMatrixArrayWithNumber(matrixArray []*Matrix, number float64)[]*Matrix{

	result := make([]*Matrix, len(matrixArray))

	for i, _ := range matrixArray{

		result[i] = MultiplyMatrixWithNumber(matrixArray[i], number)
	}
	return result

}

func MultiplyMatrixWithNumber(matrix *Matrix, number float64)*Matrix{
	numbers := make([][]float64, matrix.NumOfRows)
	for i := 0; i < matrix.NumOfRows; i++ {
		numbers[i] = make([]float64, matrix.NumOfColumns)
	}
	var result = &Matrix{NumOfRows: matrix.NumOfRows, NumOfColumns: matrix.NumOfColumns, Numbers: numbers}

	for i:=0; i<matrix.NumOfRows; i++{
		for j:=0; j<matrix.NumOfColumns; j++{

			result.Numbers[i][j] = (number * matrix.Numbers[i][j])
		}
	}

	return result

}

func CreateZerofiedMatrices(values []*Matrix)[]*Matrix{
	result := make([]*Matrix, len(values))

	for j:=0; j<len(values); j++{
		numbers := make([][]float64, values[j].NumOfRows)
		for i := 0; i < values[j].NumOfRows; i++ {
			numbers[i] = make([]float64, values[j].NumOfColumns)
		}
		result[j] = &Matrix{NumOfRows: values[j].NumOfRows, NumOfColumns: values[j].NumOfColumns, Numbers: numbers}
	}

	return result
}

func CreateZerofiedDoubleArray(values [][]float64)[][]float64{
	result := make([][]float64, len(values))
	for i := 0; i < len(values); i++ {
		result[i] = make([]float64, len(values[i]))
	}
	return result
}

func CreateArrayWithDefaultValue(length int, value float64)[]float64{
	result := make([]float64, length)
	for i := 0; i < length; i++ {
		result[i] = value
	}
	return result
}

func MultiplyVectorWithTranspose(vector1, vector2 []float64)*Matrix{

	createMatrixFromVector(vector1)

	return MultiplyMatrices(createMatrixFromVector(vector1), createTransposeMatrixFromVector(vector2))



}
func createTransposeMatrixFromVector(vector []float64)*Matrix{
	numbers := make([][]float64, 1)
	numbers[0] = make([]float64, len(vector))
	for i := 0; i < len(vector); i++ {
		numbers[0][i] = vector[i]
	}
	matrix :=&Matrix{NumOfRows:1, NumOfColumns: len(vector), Numbers: numbers}
	return matrix
}

func TransposeMatrix(matrix *Matrix)*Matrix{
	numbers := make([][]float64, matrix.NumOfColumns)
	for i := 0; i < matrix.NumOfColumns; i++ {
		numbers[i] = make([]float64, matrix.NumOfRows)
		for j:=0; j<matrix.NumOfRows;j++{
			numbers[i][j] = matrix.Numbers[j][i]
		}

	}
	var result = &Matrix{NumOfRows: matrix.NumOfColumns, NumOfColumns: matrix.NumOfRows, Numbers: numbers}

	return result
}

func GetIndexOfMaxValue(vector []float64)int{
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
