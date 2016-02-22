/*
This is the file loader. You can load the training data and labels into it by doing the following:

	path_imgs := `c:\Programing\feedforward-neural-network\data\t10k-images.idx3-ubyte`
	path_labels := `c:\Programing\feedforward-neural-network\data\t10k-labels.idx1-ubyte`

	data_handle := Loader.Create{
		Filepath_image:path_imgs,
		Filepath_labels:path_labels,
	}

	data := data_handle.Load()
	fmt.Println(data[1].Label)   // label of example 2, type uint16
	fmt.Println(data[1].Picture) // pointer to image 2, which is a slice of []byte. Each element is an 8bit integer, value between 0 and 255.
	fmt.Println(data_handle.Header_img.Magic_number) //

Data structure:
Create stores image file and label file headers after Load() has been invoked. Check below to see how to access specific items.

*/

package Loader

import (
	"os"
	"fmt"
	"encoding/binary"
	"io"

)

type Create struct {
	Filepath_image	string
	Filepath_labels	string
	Header_img	Image_header
	Header_lbl	Label_header
}

type Image_header struct {
	Magic_number	uint32
	Num_of_items	uint32
	Num_of_rows	uint32
	Num_of_columns	uint32
}

type Label_header struct {
	Magic_number	uint32
	Num_of_items	uint32
}

type training_sample struct {
	Picture *[]byte
	Label	uint16
}


// load the 2 files and return them in :: training data []training_sample ::
// I make Load() receive the reference because I would like to change the original Create struct, specifically add its image and label headers
func (c *Create) Load() []training_sample {

	// PART 1 - Open two files
	file_image, err := os.Open(c.Filepath_image)  // file is a pointer to the file, or *File
	if err != nil {
		fmt.Sprintf("File failed to be opened. Error: %v", err)
		panic(err)
	}

	file_label, err := os.Open(c.Filepath_labels)  // file is a pointer to the file, or *File
	if err != nil {
		fmt.Sprintf("File failed to be opened. Error: %v", err)
		panic(err)
	}

	defer file_image.Close()
	defer file_label.Close()

	// Image HEADER
	img_header := make([]byte, 16)
	file_image.Read(img_header);

	c.Header_img.Magic_number 	= binary.BigEndian.Uint32(img_header[0:4])
	c.Header_img.Num_of_items 	= binary.BigEndian.Uint32(img_header[4:8])
	c.Header_img.Num_of_rows 	= binary.BigEndian.Uint32(img_header[8:12])
	c.Header_img.Num_of_columns	= binary.BigEndian.Uint32(img_header[12:16])

	fmt.Println(c.Header_img.Num_of_items)
	//var training_data []byte -- make a []slice for all the training examples
	training_data := make([]training_sample, c.Header_img.Num_of_items + 1)

	// Label HEADER
	label_header := make([]byte,8)
	file_label.Read(label_header)

	c.Header_lbl.Magic_number 	= binary.BigEndian.Uint32(label_header[0:4])
	c.Header_lbl.Num_of_items	= binary.BigEndian.Uint32(label_header[4:8])

	//
	pixls 	:= c.Header_img.Num_of_rows * c.Header_img.Num_of_columns // number of pixels
	image	:= make([]byte, pixls) //

	// read all the labels into one big []byte
	labels := make([]byte, c.Header_lbl.Num_of_items + 1)
	file_label.ReadAt(labels,8)

	// Now loop through the data and populate the training_data
	// Let's loop through image data, cause it's bigger
	var j int //j will be the position of an image in training data - initialized at 0 by default

	// infinite loop. Exits when the EOF is reaches
	for i := 16; i >= 0; i = i + int(pixls) {

		count, err := file_image.ReadAt(image, int64(i))
		lbl := uint16(labels[j])

		// populate training_data with training_sample structs
		ts := training_sample{ &image, lbl }
		training_data[j] = ts
		j++

		//
		if err == io.EOF {
			fmt.Println("File read successfully.")
			fmt.Println(count)
			break
		}
	}
	return training_data
}



