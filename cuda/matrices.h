void matrices(void);
struct Matrix{//typedef
                     float* numbers;
                 	int numOfColumns;
                 	int numOfRows;
                 };
void multiplyMatrices(struct Matrix *matrix1, struct Matrix *matrix2, struct Matrix *resultMatrix);
void multiplyMatrices2(struct Matrix *matrix1, struct Matrix *matrix2, struct Matrix *resultMatrix);

float* allocArray(int ln, float* values) ;
void freeArr(float* p);
float* allocEmptyArray(int ln);
void getNumbers(float *filled, float* empty, int numOfRows, int numOfColumns);
