
struct Matrix{//typedef
                     float* numbers;
                 	int numOfColumns;
                 	int numOfRows;
                 };

void multiplyMatrices(struct Matrix *matrix1, struct Matrix *matrix2, struct Matrix *resultMatrix);
void multiplyMatrixWithVector(struct Matrix *matrix, float *vector, float *resultVector);
void applySigmoidOnVector(float *vector,float *resultVector, int size);
void applySigmoidDerivativeOnVector(float *vector,float *resultVector, int size);


float* allocArray(int ln, float* values) ;
void freeArr(float* p);
float* allocEmptyArray(int ln);
void getNumbers(float *filled, float* empty, int numOfRows, int numOfColumns);
