package cuda

import (
	//"fmt"
	"Go-Feedforward-Neural-Network/algebra"
	//"unsafe"
	"unsafe"
	"fmt"
)

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L. -lmatrices
//#cgo LDFLAGS: -lcudart -lcublas -lcurand
//#include <matrices.h>
import "C"


/**
 * compile with following instruction
 */
//nvcc -m64 -arch=sm_20 -o libmatrices.so --shared -Xcompiler -fPIC matrices.cu

func MultiplyMatrices(matrix1, matrix2 *algebra.Matrix){


	cMatrix1 := createCMatrix(matrix1)
	cMatrix2 := createCMatrix(matrix2)

	C.multiplyMatrices(cMatrix1, cMatrix2)

	cpuResult := algebra.MultiplyMatrices(matrix1, matrix2)
	fmt.Printf("\n CPU result: ",cpuResult)
	fmt.Println("")


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

