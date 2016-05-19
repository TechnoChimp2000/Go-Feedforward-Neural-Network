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

	C.multiplyMatrices2(cMatrix1, cMatrix2, cResultMatrix)

	goResultMatrix := createGoMatrix(cResultMatrix)


	return goResultMatrix


}

func MatricesMultiplication() {
	/*C.matrices()

	fmt.Println("Done multiplying matrices")*/

}

func createCMatrix(matrix *algebra.Matrix)*C.struct_Matrix{
	array := algebra.ReturnMatrixInSingleArray(matrix)
	cArray := C.allocArray((C.int)(len(array)), (*C.float)(&array[0]))

	cMatrix := C.struct_Matrix{cArray, (C.int)(matrix.NumOfColumns), (C.int)(matrix.NumOfRows)}

	return (*C.struct_Matrix)(unsafe.Pointer(&cMatrix));

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

