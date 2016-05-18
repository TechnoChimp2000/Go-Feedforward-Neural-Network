package network2

func convertArray(array []float32)[]float64{
	result := make([]float64, len(array))
	for i,_ := range array{
		result[i] = (float64)(array[i])
	}
	return result
}
