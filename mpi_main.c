#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
//#define N 10000

float maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
float w = 0.5;
float eps;
int mpi_size = 0, mpi_rank = 0;

float A [N][N][N];
float B [N][N][N];
MPI_Status status;

void memoryAllocate(float ****, int);
void freeMemory(float ****);
void relax(int, int);
void init();
void verify();
void checkForSuccess(int, int); 

int main(int an, char **as)
{
	int it, ibeg, iend, count;
    double start, end;  
    checkForSuccess(MPI_Init(&an, &as), 1);
    checkForSuccess(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size), 2);
    checkForSuccess(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank), 3);

    start = MPI_Wtime();

    ibeg = mpi_rank * N / mpi_size;
    if(mpi_rank == mpi_size - 1)
        iend = N - 1;
    else
        iend = (mpi_rank + 1) * N / mpi_size - 1;

	init();

    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax(ibeg, iend);
        if(mpi_rank == 0)
            printf( "it=%4i   eps=%f\n", it,eps);
        if (eps < maxeps) break;
    }
    if(mpi_rank == 0)
        verify();
    end = MPI_Wtime();
    printf("work time %f\n", end - start);
    MPI_Finalize();
	return 0;
}


void init()
{ 
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)     A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 


void relax(int ibeg, int iend)
{
	int count = 0, iTempBeg, iTempEnd, ii;
    float tempEps;

	for(i=ibeg; i<=iend; i++)
    {
        if(i > N - 2 || i < 1)
            continue;
        for(j=1; j<=N-2; j++)
	        for(k=1+(i+j)%2; k<=N-2; k+=2)
            { 
                float b;
                b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
                +A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);
                eps =  Max(fabs(b),eps);
                A[i][j][k] = A[i][j][k] + b;
            }
    }
	  
    // Send/Recv
    if(mpi_rank != 0)
    {
        checkForSuccess(MPI_Send(&eps, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD), 4);
        checkForSuccess(MPI_Send(&ibeg, 1, MPI_INT, 0, 1, MPI_COMM_WORLD), 4);
        checkForSuccess(MPI_Send(&iend, 1, MPI_INT, 0, 1, MPI_COMM_WORLD), 4);
        checkForSuccess(MPI_Send(A, N * N * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD), 4);
    }
    else
    {
        for(i = 1; i < mpi_size; i++)
        {
            checkForSuccess(MPI_Recv(&tempEps, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status), 5);
            checkForSuccess(MPI_Recv(&iTempBeg, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status), 5);
            checkForSuccess(MPI_Recv(&iTempEnd, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status), 5);
            checkForSuccess(MPI_Recv(B, N * N * N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status), 5);
            for(ii = iTempBeg; ii <= iTempEnd; ii++)
                for(j=0; j<=N-1; j++)
	                for(k=0; k<=N-1; k++)
                        A[ii][j][k] = B[ii][j][k];
            if(tempEps > eps)
                eps = tempEps;
        }
    }
    checkForSuccess(MPI_Bcast(A, N*N*N, MPI_FLOAT, 0, MPI_COMM_WORLD), 6);

	for(i=ibeg; i<=iend; i++)
    {
        if(i > N - 2 || i < 1)
            continue;
	    for(j=1; j<=N-2; j++)
	        for(k=1+(i+j+1)%2; k<=N-2; k++=2)
            {
                float b;
                b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
                +A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
            
            }
    }

    if(mpi_rank != 0)
    { 
        checkForSuccess(MPI_Send(&ibeg, 1, MPI_INT, 0, 1, MPI_COMM_WORLD), 4);
        checkForSuccess(MPI_Send(&iend, 1, MPI_INT, 0, 1, MPI_COMM_WORLD), 4);
        checkForSuccess(MPI_Send(A, N * N * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD), 4);
    }
    else
    {
        for(i = 1; i < mpi_size; i++)
        {
            checkForSuccess(MPI_Recv(&iTempBeg, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status), 5);
            checkForSuccess(MPI_Recv(&iTempEnd, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status), 5);
            checkForSuccess(MPI_Recv(B, N * N * N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status), 5);

            for(ii = iTempBeg; ii <= iTempEnd; ii++)
                for(j=0; j<=N-1; j++)
	                for(k=0; k<=N-1; k++)
                        A[ii][j][k] = B[ii][j][k];
        }
    }
    checkForSuccess(MPI_Bcast(A, N*N*N, MPI_FLOAT, 0, MPI_COMM_WORLD), 6);
}


void verify()
{ 
	float s;

	s=0.;
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		int l = A[i][j][k];
		l = l * (i+1);
		l = l / N;
		l = l * (j+1);
		l = l / N;
		l = l * (k+1);
		l /= N;
		s=s+l;
	}
	printf("  S = %f\n",s);

}

void checkForSuccess(int rc, int errCode)
{
    if(rc != MPI_SUCCESS)
    {
        fprintf(stderr, "Error calling MPI function!\n");
        MPI_Abort(MPI_COMM_WORLD, errCode);
    }
}



