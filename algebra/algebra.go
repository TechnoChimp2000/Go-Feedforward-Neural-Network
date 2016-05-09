package algebra

import (
"math"
"math/rand"
)

type Matrix struct{
	numbers [][]float32
	numOfColumns int
	numOfRows int
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

func CreateNormalizedMatrix(numOfRows int, numOfColumns int) *Matrix{
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

}

/**
* Z-scores normalized vector has mean 0 and standard deviation 1
*/
func CreateNormalizedVector(length int) []float32{
	vector := make([]float32, length)
	/**
	 * randomize vector first
	 */
	for i:=0; i<length; i++{
		vector[i] = rand.Float32()
	}


	var mean float32
	for _, value := range vector {
		mean = mean + value
	}

	mean = mean/float32(len(vector))

	var deviation float32

	if len(vector) > 1 {


		for _, value := range vector{
			deviation = deviation + ((value - mean) * (value - mean)) / float32(len(vector) - 1)
		}

		deviation = float32(math.Sqrt(float64(deviation)))

	}

	var result =  make([]float32, len(vector))

	if len(vector) > 1 && deviation!= 0{

		//fmt.Println("(1.1) Vector has value {1}", vector)
		for index, _ := range vector {
			result[index] = (vector[index] - mean) / deviation
		}
		//fmt.Println("(2.1) Normalized has value {1}", result)

	}else{
		return vector
	}
	return result

}

func Vectorize(function func(float32)float32, vector []float32) []float32{
	result := make([]float32, len(vector))

	for i, value := range vector{
		result[i] = function(value)
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
