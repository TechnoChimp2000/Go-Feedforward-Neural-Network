package cuda

import (
	//"fmt"
	"Go-Feedforward-Neural-Network/algebra"
	//"unsafe"
	"unsafe"
	//"fmt"
)

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L. -lmatrices
//#cgo LDFLAGS: -lcudart -lcublas -lcurand
//#include <matrices.h>
//#include <stdio.h>
//#include <stdlib.h>
import "C"


/**
 * compile with following instruction
 */
//nvcc -m64 -arch=sm_20 -o libmatrices.so --shared -Xcompiler -fPIC matrices.cu

func MultiplyMatrices(matrix1, matrix2 *algebra.Matrix)*algebra.Matrix{

	//C.free(unsafe.Pointer(cs))


	cMatrix1 := createCMatrix(matrix1)
	//defer C.free(unsafe.Pointer(cMatrix1))

	cMatrix2 := createCMatrix(matrix2)
	//defer C.free(unsafe.Pointer(cMatrix2))

	cResultMatrix := initializeResultCMatrix(matrix1, matrix2)

	C.multiplyMatrices(cMatrix1, cMatrix2, cResultMatrix)

	goResultMatrix := createGoMatrix(cResultMatrix)


	return goResultMatrix


}

func MultiplyMatrixWithVector(matrix *algebra.Matrix, vector []float32)[]float32{
	cMatrix := createCMatrix(matrix)
	cVector := createCVector(vector)
	cResult := createEmptyCVector(len(vector))

	C.multiplyMatrixWithVector(cMatrix, cVector, cResult)

	return createGoVector(cResult, matrix.NumOfRows)

}




func createCMatrix(matrix *algebra.Matrix)*C.struct_Matrix{
	array := algebra.ReturnMatrixInSingleArray(matrix)
	cArray := C.allocArray((C.int)(len(array)), (*C.float)(&array[0]))

	cMatrix := C.struct_Matrix{cArray, (C.int)(matrix.NumOfColumns), (C.int)(matrix.NumOfRows)}

	return (*C.struct_Matrix)(unsafe.Pointer(&cMatrix));

}

func createGoVector(cVector *C.float, length int)[]float32{
	goNumbers :=  make([]float32, length) //(int)(resultNumOfRows) * (int)(resultNumOfColumns)
	C.getNumbers(cVector, (*C.float)(&goNumbers[0]), (C.int)(length), 1)
	return goNumbers
}

func createCVector(goVector []float32)*C.float{
	return C.allocArray((C.int)(len(goVector)), (*C.float)(&goVector[0]))
}

func createEmptyCVector(length int)*C.float{
	return C.allocEmptyArray((C.int)(length))
}

func createGoMatrix(cMatrix *C.struct_Matrix)*algebra.Matrix{
	//defer C.free(unsafe.Pointer(cMatrix))

	resultNumOfRows := cMatrix.numOfRows


	resultNumOfColumns:= cMatrix.numOfColumns


	resultNumbers:= cMatrix.numbers
	goNumbers :=  make([]float32, (int)(resultNumOfRows) * (int)(resultNumOfColumns)) //(int)(resultNumOfRows) * (int)(resultNumOfColumns)
	C.getNumbers(resultNumbers, (*C.float)(&goNumbers[0]), resultNumOfRows, resultNumOfColumns)






	return algebra.ReturnMatrixFromSingleArray((int)(resultNumOfRows), (int)(resultNumOfColumns), goNumbers)
}

func initializeResultCMatrix(matrix1, matrix2 *algebra.Matrix)*C.struct_Matrix{

	cArray := C.allocEmptyArray((C.int)(matrix1.NumOfRows * matrix2.NumOfColumns))

	cMatrix := C.struct_Matrix{cArray, (C.int)(matrix2.NumOfColumns), (C.int)(matrix1.NumOfRows)}

	return (*C.struct_Matrix)(unsafe.Pointer(&cMatrix));

}

