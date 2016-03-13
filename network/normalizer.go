package network

import (
	//"fmt"
	"math"
)

type Normalizer interface{
	normalizeVector(vector []float32) []float32
}



func normalizeTrainingInput(trainingSet []TrainingSample, normalizer Normalizer){
	for _, trainingSample := range trainingSet {
		normalizedVector := normalizer.normalizeVector(trainingSample.Input)
		for index, _ := range trainingSample.Input {

			trainingSample.Input[index] = normalizedVector[index]

		}
	}

}

type ZscoreNormalizer struct {}

/**
 * Z-scores normalized vector has mean 0 and standard deviation 1
 */
func(n *ZscoreNormalizer)normalizeVector(vector []float32) []float32{
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

type SkipNormalizer struct {}

/**
 * doesn't normalize, just returns input
 */
func(s *SkipNormalizer)normalizeVector(vector []float32) []float32{
	return vector
}


