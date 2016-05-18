package network2


type Regularization interface{
	getRegularizationFactor(trainingSetSize int, eta float64)float64
}

type L2Regularization struct{
	lambda float64
}

func(l *L2Regularization)getRegularizationFactor(trainingSetSize int, eta float64)float64{
	//(1-eta*(lmbda/n)
	return 1 - (eta*(l.lambda/(float64)(trainingSetSize)))
}

type SkipRegularization struct{}

func(s *SkipRegularization)getRegularizationFactor(trainingSetSize int, eta float64)float64{
	return 1
}


