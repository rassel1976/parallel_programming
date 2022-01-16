#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
//#define N 10000
float   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
float w = 0.5;
float eps;

float A [N][N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;

	double start, end, exec;

	start = omp_get_wtime();

	omp_set_num_threads(4);

	init();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();

	end = omp_get_wtime();
	exec = end - start;
	printf("start time &lf\nend time %lf\n exec time %lf\n", start, end, exec);

	return 0;
}


void init()
{ 
	#pragma omp parallel for shared(A) private(i, j, k)
	for(k=0; k<=N-1; k++)
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)     A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 


void relax()
{
	
	#pragma omp parallel for shared(A, eps) private(i, j, k) reduction(max:eps) 
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1+(k+j)%2; i<=N-2; i+=2)
	{ 
		float b;
		b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
		+A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);
		eps = fmaxf(fabs(b), eps);
		A[i][j][k] = A[i][j][k] + b;
	}
	
	#pragma omp parallel for shared(A) private(i, j, k)
	for(k=1; k<=N-2; k++)
	for(j=1; j<=N-2; j++)
	for(i=1+(k+j+1)%2; i<=N-2; i+=2)
	{
		float b;
		b = w*((A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]
		+A[i][j][k-1]+A[i][j][k+1] )/6. - A[i][j][k]);
		A[i][j][k] = A[i][j][k] + b;
	}

}


void verify()
{ 
	float s;

	s=0.;
	#pragma omp parallel for shared(A) private(i, j, k) reduction(+:s)
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


