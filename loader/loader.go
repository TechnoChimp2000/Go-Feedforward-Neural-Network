package loader

import (
	"io/ioutil"
	"fmt"
	"encoding/binary"

	"github.com/Go-Feedforward-Neural-Network/network"
)

// this is a generic IDX loader that reads a file and returns it in a slice. Example of operation:
/*

path_images_train := `c:\Programing\feedforward-neural-network\data\train-images.idx3-ubyte`
path_labels_train := `c:\Programing\feedforward-neural-network\data\train-labels.idx1-ubyte`

dh_images := &loader.Create{
	Filepath:path_images_train,
}

dh_labels := &loader.Create{
	Filepath:path_labels_train,
}

dh_images.Load()
dh_labels.Load()

fmt.Println("data images header: ", dh_images.Header);
fmt.Println("data length: ", len(dh_images.Data));
fmt.Println("data point length: ", len(dh_images.Data[0]));

fmt.Println("data labels header: ", dh_labels.Header);
fmt.Println("data length: ", len(dh_labels.Data));
fmt.Println("data point length: ", len(dh_labels.Data[0]));

digit_count	:= uint32(10)
train_data 	:= loader.CreateTrainingSet( dh_images, dh_labels , digit_count)

*/
// data structures
type Create struct {
	Filepath string
	Header   Header
	Data	[]DataPoint
}

type Header struct {
	Magic_number		byte
	Num_of_dimensions	int
	Num_of_items		[]uint32
}

type DataPoint []uint32

type TrainingSample_loader struct {
	Input 	[]uint32
	Output	[]uint32
}

// functions
func (c *Create) Load() ( output float64 ) {

	dh, err := ioutil.ReadFile( c.Filepath )
	if err != nil {
		fmt.Sprintf("File failed to be opened. Error: %v", err)
		panic(err)
	}

	// Read first four bytes for the header that have to be 0
	//0-1
	if dh[0] != 0 || dh[1] != 0 {
		panic(err)
	}

	//2-3
	c.Header.Magic_number		= dh[2]
	c.Header.Num_of_dimensions 	= int(dh[3])

	c.Header.Num_of_items = make([]uint32, c.Header.Num_of_dimensions)

	// collect number of dimensions and store them into a slice as part of the header
	for i := 0; i< c.Header.Num_of_dimensions; i++ {

		offset          := 4
		start_index	:= offset + i*4
		end_index	:= offset + start_index

		fmt.Println( binary.BigEndian.Uint32(dh[start_index:end_index]) )
		c.Header.Num_of_items[i] = binary.BigEndian.Uint32(dh[start_index:end_index])

	}

	// process data until the end of file
	offset_body := 4 + c.Header.Num_of_dimensions * 4
	//fmt.Println("body offset", offset_body)

	size := dataSize(c.Header.Num_of_items)
	//fmt.Println("size: ", size)

	var data_point DataPoint
	pixel_counter 	:= uint32(0)

	var dp = make([]uint32, size)
	//fmt.Sprintf("%T ", data_point)

	for i := range dh {

		// skip the header
		if i < offset_body {
			continue
		}

		data_point 		= append(data_point, uint32(dh[i]))
		dp[pixel_counter] 	= uint32(dh[i])
		pixel_counter 		+= 1

		// at the end of each item, store it to c.Data
		if pixel_counter == size {
			c.Data 		= append(c.Data, data_point)
			pixel_counter 	= uint32(0)
			data_point 	= nil
		}
	}
	return output
}

// if needed we can in the future map the magic number to a stated type
func (c *Create ) MagicNumberMap() {

}

// you feed it a slice, it skips the first element, and multiplies the rest with itself to get the total number of elements in the data point
func dataSize( num_of_items []uint32  ) ( size uint32 ) {
	size = 1

	for i := 1; i<len(num_of_items); i++ {
		size *= num_of_items[i]
	}
	return size
}



func CreateTrainingSet ( X *Create, Y *Create, digit_count uint32 ) ( training_data []network.TrainingSample ) {
	// validate the two sets ( do they have the same number of items )

	if X.Header.Num_of_items[0] != Y.Header.Num_of_items[0] {
		panic("Number of items mismatch")
	}

	// loop
	training_data = make([]network.TrainingSample, X.Header.Num_of_items[0])
	//training_data = make([]network.TrainingSample, 200)
	// size of X

	xSize := 784 // TODO: make this general




	for i := range training_data {

		var Y_normal = LabelNormalization( Y.Data[i][0], digit_count )

		var X_normal []float32
		X_normal = make([]float32, xSize)

		for index, value := range X.Data[i] {
			X_normal[index] = float32(value)
		}
		//fmt.Println(len(X_normal))

		training_data[i]=network.TrainingSample{Input:X_normal, Output:Y_normal}
	}

	return training_data
}

func LabelNormalization( input, label_count uint32 ) (vector []float32){
	//
	vector = make( []float32, label_count )

	if input == 0 {
		vector[label_count-1] = 1
	} else {
		vector[input-1] = 1
	}

	return vector
}




