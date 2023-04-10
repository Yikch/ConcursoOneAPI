#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define NTHREADS 25
#define TILE_DIM 29  //TILE_DIM = NTHREADS + 4

__global__ void gpu_canny_nr(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	__shared__ float tile [ TILE_DIM ] [ TILE_DIM ]; 
	i = blockIdx.y * NTHREADS + threadIdx.y; 
	j = blockIdx.x * NTHREADS + threadIdx.x; 
	if(i<height && j<width){
		tile[threadIdx.y][threadIdx.x] = im[i*width+j];
		__syncthreads();
		if(i>=2 && i<height-2 && threadIdx.x>=2 && threadIdx.x<TILE_DIM-2 && j >=2 && j<width-2 && threadIdx.y>=2 && threadIdx.y<TILE_DIM-2){
			// Noise reduction
			NR[i*width+j] =
				(2.0*tile[threadIdx.y-2][threadIdx.x-2]	 +  4.0*tile[threadIdx.y-2][threadIdx.x-1] +  5.0*tile[threadIdx.y-2][threadIdx.x] +  4.0*tile[threadIdx.y-2][threadIdx.x+1] + 2.0*tile[threadIdx.y-2][threadIdx.x+2]
				+ 4.0*tile[threadIdx.y-1][threadIdx.x-2] +  9.0*tile[threadIdx.y-1][threadIdx.x-1] + 12.0*tile[threadIdx.y-1][threadIdx.x] +  9.0*tile[threadIdx.y-1][threadIdx.x+1] + 4.0*tile[threadIdx.y-1][threadIdx.x+2]
				+ 5.0*tile[threadIdx.y  ][threadIdx.x-2] + 12.0*tile[threadIdx.y  ][threadIdx.x-1] + 15.0*tile[threadIdx.y  ][threadIdx.x] + 12.0*tile[threadIdx.y  ][threadIdx.x+1] + 5.0*tile[threadIdx.y][threadIdx.x+2]
				+ 4.0*tile[threadIdx.y+1][threadIdx.x-2] +  9.0*tile[threadIdx.y+1][threadIdx.x-1] + 12.0*tile[threadIdx.y+1][threadIdx.x] +  9.0*tile[threadIdx.y+1][threadIdx.x+1] + 4.0*tile[threadIdx.y+1][threadIdx.x+2]
				+ 2.0*tile[threadIdx.y+2][threadIdx.x-2] +  4.0*tile[threadIdx.y+2][threadIdx.x-1] +  5.0*tile[threadIdx.y+2][threadIdx.x] +  4.0*tile[threadIdx.y+2][threadIdx.x+1] + 2.0*tile[threadIdx.y+2][threadIdx.x+2])
				/159.0;
		}
	}

}

__global__ void gpu_canny_gradient(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	__shared__ float tile [ TILE_DIM ] [ TILE_DIM ]; 
	float PI = 3.141593;
	i = blockIdx.y * NTHREADS + threadIdx.y; 
	j = blockIdx.x * NTHREADS + threadIdx.x; 
	if(i<height && j<width){
		tile[threadIdx.y][threadIdx.x] = NR[i*width+j];
		__syncthreads();
		if(i>=2 && i<height-2 && threadIdx.x>=2 && threadIdx.x<TILE_DIM-2 && j >=2 && j<width-2 && threadIdx.y>=2 && threadIdx.y<TILE_DIM-2){
			// Intensity gradient of the image
			Gx[i*width+j] = 
				 (1.0*tile[threadIdx.y-2][threadIdx.x-2] +  2.0*tile[threadIdx.y-2][threadIdx.x-1] +  (-2.0)*tile[threadIdx.y-2][threadIdx.x+1] + (-1.0)*tile[threadIdx.y-2][threadIdx.x+2]
				+ 4.0*tile[threadIdx.y-1][threadIdx.x-2] +  8.0*tile[threadIdx.y-1][threadIdx.x-1] +  (-8.0)*tile[threadIdx.y-1][threadIdx.x+1] + (-4.0)*tile[threadIdx.y-1][threadIdx.x+2]
				+ 6.0*tile[threadIdx.y  ][threadIdx.x-2] + 12.0*tile[threadIdx.y  ][threadIdx.x-1] + (-12.0)*tile[threadIdx.y  ][threadIdx.x+1] + (-6.0)*tile[threadIdx.y  ][threadIdx.x+2]
				+ 4.0*tile[threadIdx.y+1][threadIdx.x-2] +  8.0*tile[threadIdx.y+1][threadIdx.x-1] +  (-8.0)*tile[threadIdx.y+1][threadIdx.x+1] + (-4.0)*tile[threadIdx.y+1][threadIdx.x+2]
				+ 1.0*tile[threadIdx.y+2][threadIdx.x-2] +  2.0*tile[threadIdx.y+2][threadIdx.x-1] +  (-2.0)*tile[threadIdx.y+2][threadIdx.x+1] + (-1.0)*tile[threadIdx.y+2][threadIdx.x+2]);


			Gy[i*width+j] = 
				 ((-1.0)*tile[threadIdx.y-2][threadIdx.x-2] + (-4.0)*tile[threadIdx.y-2][threadIdx.x-1] +  (-6.0)*tile[threadIdx.y-2][threadIdx.x] + (-4.0)*tile[threadIdx.y-2][threadIdx.x+1] + (-1.0)*tile[threadIdx.y-2][threadIdx.x+2]
				+ (-2.0)*tile[threadIdx.y-1][threadIdx.x-2] + (-8.0)*tile[threadIdx.y-1][threadIdx.x-1] + (-12.0)*tile[threadIdx.y-1][threadIdx.x] + (-8.0)*tile[threadIdx.y-1][threadIdx.x+1] + (-2.0)*tile[threadIdx.y-1][threadIdx.x+2]
				+    2.0*tile[threadIdx.y+1][threadIdx.x-2] +    8.0*tile[threadIdx.y+1][threadIdx.x-1] +    12.0*tile[threadIdx.y+1][threadIdx.x] +    8.0*tile[threadIdx.y+1][threadIdx.x+1] +    2.0*tile[threadIdx.y+1][threadIdx.x+2]
				+    1.0*tile[threadIdx.y+2][threadIdx.x-2] +    4.0*tile[threadIdx.y+2][threadIdx.x-1] +     6.0*tile[threadIdx.y+2][threadIdx.x] +    4.0*tile[threadIdx.y+2][threadIdx.x+1] +    1.0*tile[threadIdx.y+2][threadIdx.x+2]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
		}
	}
}

__global__ void gpu_canny_edge(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	__shared__ float tile [ TILE_DIM ] [ TILE_DIM ]; 
	i = blockIdx.y * NTHREADS + threadIdx.y; 
	j = blockIdx.x * NTHREADS + threadIdx.x; 
	if(i<height && j<width){
		tile[threadIdx.y][threadIdx.x] = G[i*width+j];
		__syncthreads();

		if(i>=3 && i<height-3 && threadIdx.x>=1 && threadIdx.x<TILE_DIM-3 && j >=3 && j<width-3 && threadIdx.y>=1 && threadIdx.y<TILE_DIM-3){
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(tile[i][j]>tile[i][j+1] && tile[i][j]>tile[i][j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(tile[i][j]>tile[i+1][j+1] && tile[i][j]>tile[i-1][j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(tile[i][j]>tile[i][j] && tile[i][j]>tile[i-1][j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(tile[i][j]>tile[i+1][j-1] && tile[i][j]>tile[i-1][j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
		}
	}
}

__global__ void gpu_canny_out(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	int ii, jj;
	float lowthres, hithres;
	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 
	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);
	if(i>=3 && i<height-3 && j >=3 && j<width-3){
		image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
	}
}


__global__ void gpu_hough_init(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int j;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j<(accu_width*accu_height)){
		accumulators[j]=0;
	}
}

__global__ void gpu_hough_exec(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int i, j, theta;
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	float center_x = width/2.0; 
	float center_y = height/2.0;
	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 
	if(i<height && j<width){
		if( im[ (i*width) + j] > 250 ) // Pixel is edge  
			{  
				for(theta=0;theta<180;theta++)  
				{  
					float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]++;

				} 
			} 
	}
}

void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

void canny(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width)
{
	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;

	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}


	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Intensity gradient of the image
			Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
	}

	// Edge
	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
		}

	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);

	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
		}
}

void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	int threshold;

	// Create temporal buffers 
	uint8_t *imEdge_CPU = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float *NR_CPU = (float *)malloc(sizeof(float) * width * height);
	float *G_CPU = (float *)malloc(sizeof(float) * width * height);
	float *phi_CPU = (float *)malloc(sizeof(float) * width * height);
	float *Gx_CPU = (float *)malloc(sizeof(float) * width * height);
	float *Gy_CPU = (float *)malloc(sizeof(float) * width * height);
	uint8_t *pedge_CPU = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	canny(im, imEdge_CPU,
		NR_CPU, G_CPU, phi_CPU, Gx_CPU, Gy_CPU, pedge_CPU,
		1000.0f, //level
		height, width);


	/* CUDA vesion */

	dim3 dimBlock(TILE_DIM,TILE_DIM);
	int blocks_h = (height)/NTHREADS;
	int blocks_w = (width)/NTHREADS;
	if ((height)%NTHREADS>0) blocks_h++;
	if ((width)%NTHREADS>0) blocks_w++;
	dim3 dimGrid(blocks_w, blocks_h);

	gpu_canny_nr<<<dimGrid,dimBlock>>>(im, imEdge, NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();
	gpu_canny_gradient<<<dimGrid,dimBlock>>>(im, imEdge, NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();
	gpu_canny_edge<<<dimGrid,dimBlock>>>(im, imEdge, NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();

	dim3 dimBlock_out(NTHREADS, NTHREADS);
	gpu_canny_out<<<dimGrid,dimBlock_out>>>(im, imEdge, NR, G, phi, Gx, Gy, pedge,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();

	//Comprobacion de canny
	int i, j, desigual;
	i=0;
	desigual=0;
	while(i<height && !desigual){
		j=0;
		while(j<width && !desigual){
			if(NR[i*width+j] != NR_CPU[i*width+j]){
				printf("NR!= : NR_GPU[%d][%d]: %f  NR_GPU[%d][%d]: %f\n", i, j, NR[i*width+j], i,j,NR_CPU[i*width+j]);
				desigual=1;
			}
			if(G[i*width+j] != G_CPU[i*width+j]){
				printf("G!= : G_GPU[%d][%d]: %f  G_GPU[%d][%d]: %f\n", i, j, G[i*width+j], i,j,G_CPU[i*width+j]);
				desigual=1;
			}
			j++;
		}
		i++;
	}

	dim3 dimBlock_init(NTHREADS);
	blocks_h = (accu_height*accu_width)/NTHREADS;
	if ((accu_height*accu_width)%NTHREADS>0) blocks_h++;
	dim3 dimGrid_init(blocks_h);
	gpu_hough_init<<<dimGrid_init,dimBlock_init>>>(imEdge, width, height, accum, accu_width, accu_height, 
		sin_table, cos_table);
	cudaDeviceSynchronize();

	dim3 dimBlock_exec(NTHREADS, NTHREADS);
	blocks_h = (height)/NTHREADS;
	blocks_w = (width)/NTHREADS;
	if ((height)%NTHREADS>0) blocks_h++;
	if ((width)%NTHREADS>0) blocks_w++;
	dim3 dimGrid_exec(blocks_w, blocks_h);
	gpu_hough_exec<<<dimGrid_exec,dimBlock_exec>>>(imEdge, width, height, accum, accu_width, accu_height, 
		sin_table, cos_table);
	cudaDeviceSynchronize();

	if (width>height) threshold = width/6;
	else threshold = height/6;
	
	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}