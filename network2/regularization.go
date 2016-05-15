package network2


type Regularization interface{
	getRegularizationFactor(trainingSetSize int, eta float32)float32
}

type L2Regularization struct{
	lambda float32
}

func(l *L2Regularization)getRegularizationFactor(trainingSetSize int, eta float32)float32{
	//(1-eta*(lmbda/n)
	return 1 - (eta*(l.lambda/(float32)(trainingSetSize)))
}

type SkipRegularization struct{}

func(s *SkipRegularization)getRegularizationFactor(trainingSetSize int, eta float32)float32{
	return 1
}


