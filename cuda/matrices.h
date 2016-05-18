void matrices(void);
struct Matrix{//typedef
                     float* numbers;
                 	int numOfColumns;
                 	int numOfRows;
                 };
void multiplyMatrices(struct Matrix *matrix1,struct Matrix *matrix2);

float* allocArray(int ln, float* values) ;
void freeArr(float* p);
