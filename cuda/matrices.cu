// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>

char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


struct sigmoid_functor
{
  __host__ __device__
  float operator()(const float& x) const
  {
       return 1/(1.0 + expf(-x));
  }
};

struct sigmoid_derivative_functor
{
  __host__ __device__
  float operator()(const float& x) const
  {
       return expf(x)/powf((1.0 + expf(x)),2);
  }
};




// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

void printArray(float *array ,int length){
     std::cout << std::endl;
      std::cout << "array: ";
    for(int i = 0; i < length; ++i){
        std::cout << array[i] << " ";
    }
}

//y = α op(A)x + βy
void gpu_blas_matrix_with_vector(const float *A, const float *v, float *result, const int m, const int n) {

	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
//	char* resultStatus;


	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);


	// Do the actual multiplication
	// matrix - vector multiplication : d_y = al*d_a *d_x + bet *d_y
    // d_a - mxn matrix ; d_x - n-vector , d_y - m- vector ;
    // al ,bet - scalars
    //cublasSgemv(handle,CUBLAS OP N,m,n,&al,d a,m,d x,1,&bet,d y,1);
    /*resultStatus = cublasGetErrorString(cublasSgemv(handle,CUBLAS_OP_N,m,n,alpha,A,m,v,1,beta,result,1));
     std::cout << "CUblas result: \n" << resultStatus;
*/
    cublasSgemv(handle,CUBLAS_OP_N,m,n,alpha,A,m,v,1,beta,result,1);

	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}




extern "C"{
    float* allocArray(int ln, float* values) {
        float* result = (float*) malloc(ln * sizeof(float));
        for (int i = 0; i < ln; i++) {
                result[i] = values[i];
        }
        return result;
    }

    void freeArr(float* p) { free(p); }

    float* allocEmptyArray(int ln) {
            float* result = (float*) malloc(ln * sizeof(float));

            return result;
    }

    void getNumbers(float *filled, float* empty, int numOfRows, int numOfColumns){
         for (int i = 0; i < (numOfRows * numOfColumns); i++) {
                        empty[i] = filled[i];
          }
    }
}


 extern "C" {
    typedef struct {//typedef
        float* numbers;
    	int numOfColumns;
    	int numOfRows;
    } Matrix;
}

void print_matrix_struct(Matrix *matrix) {
     for(int i = 0; i < (matrix->numOfColumns * matrix->numOfRows); ++i){
        std::cout << matrix->numbers[i] << " ";

     }
     std::cout << std::endl;
}

 extern "C" {

        void multiplyMatrices(Matrix *matrix1, Matrix *matrix2, Matrix *resultMatrix) {

                    thrust::device_vector<float> A(matrix1->numbers, matrix1->numbers+(matrix1->numOfRows * matrix1->numOfColumns));
                    thrust::device_vector<float> B(matrix2->numbers, matrix2->numbers+(matrix2->numOfRows * matrix2->numOfColumns));
                    thrust::device_vector<float> C(matrix1->numOfRows * matrix2->numOfColumns);


                    // Multiply A and B on GPU
                    gpu_blas_mmul(thrust::raw_pointer_cast(&A[0]), thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]), matrix1->numOfRows, matrix1->numOfColumns, matrix2->numOfColumns);


                    thrust::copy(C.begin(), C.end(), resultMatrix->numbers );

                    resultMatrix->numOfRows = matrix1->numOfRows;
                    resultMatrix->numOfColumns = matrix2->numOfColumns;

         }

         void multiplyMatrixWithVector(Matrix *matrix, float *vector, float *resultVector){

             thrust::device_vector<float> A(matrix->numbers, matrix->numbers+(matrix->numOfRows * matrix->numOfColumns));
             thrust::device_vector<float> v(vector, vector+matrix->numOfColumns);
             thrust::device_vector<float> result(matrix->numOfRows);

             gpu_blas_matrix_with_vector(thrust::raw_pointer_cast(&A[0]), thrust::raw_pointer_cast(&v[0]), thrust::raw_pointer_cast(&result[0]), matrix->numOfRows, matrix->numOfColumns);//matrix->numOfRows, matrix->numOfColumns

             thrust::copy(result.begin(), result.end(), resultVector);

         }

         void applySigmoidOnVector(float *vector,float *resultVector, int size){
             thrust::device_vector<float> v(vector, vector+size);
             thrust::device_vector<float> result(size);

             thrust::transform(v.begin(), v.end(), result.begin(), sigmoid_functor());

             thrust::copy(result.begin(), result.end(), resultVector);
         }

         void applySigmoidDerivativeOnVector(float *vector,float *resultVector, int size){
              thrust::device_vector<float> v(vector, vector+size);
              thrust::device_vector<float> result(size);

              thrust::transform(v.begin(), v.end(), result.begin(), sigmoid_derivative_functor());

              thrust::copy(result.begin(), result.end(), resultVector);
         }
}