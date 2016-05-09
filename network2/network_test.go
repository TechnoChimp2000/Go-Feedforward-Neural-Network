package network2

import (
	"testing"
	"fmt"
)

func TestFeedforward(t *testing.T){
	topology := []int{4,3,5, 6, 7}
	network := CreateNetwork(topology)

	input := []float32{3.14, 6, 7, 8, 8, 9}
	result := network.Feedforward(input)
	fmt.Printf("Result: ", result)
}
