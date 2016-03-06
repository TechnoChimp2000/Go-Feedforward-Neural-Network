package cuda

import "fmt"

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L. -ltest
//#cgo LDFLAGS: -lcudart
//#include <test.h>
import "C"


/**
 * compile with following instruction
 */
//nvcc -m64 -arch=sm_20 -o libtest.so --shared -Xcompiler -fPIC test.cu



func TestAddition() {

	fmt.Println("Addition is ", C.test_addition() )

}

func TestCountDevices() {

	C.count_devices()
}
