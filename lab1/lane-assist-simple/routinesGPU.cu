#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define NTHREADS 16

__global__ void gpu_canny_nr(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 
	if(i>=2 && i<height-2 && j >=2 && j<width-2){
		// Noise reduction
		NR[i*width+j] =
				(2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
			+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
			+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
			+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
			+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
			/159.0;
	}

}

__global__ void gpu_canny_gradient(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	float PI = 3.141593;
	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 
	if(i<height && j<width){
		if(i>=2 && i<height-2 && j >=2 && j<width-2){
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
	}
}

__global__ void gpu_canny_edge(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level, int height, int width)
{
	int i, j; 
	i = blockIdx.y * blockDim.y + threadIdx.y; 
	j = blockIdx.x * blockDim.x + threadIdx.x; 
	if(i<height && j<width){
		if(i>=3 && i<height-3 && j >=3 && j<width-3){
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
				for(ii=-4;ii<=4;ii++)  
				{  
					for(jj=-4;jj<=4;jj++)  
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

//Houghtransform en CPU porque accede a accumulators de forma que puede hacer accesos 
// simultaneos a memoria que causa error
void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int i, j, theta;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	for(i=0; i<accu_width*accu_height; i++)
		accumulators[i]=0;	

	float center_x = width/2.0; 
	float center_y = height/2.0;
	for(i=0;i<height;i++)  
	{  
		for(j=0;j<width;j++)  
		{  
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
}

void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	int threshold;

	/* CUDA vesion */

	uint8_t *im_GPU;
	uint8_t *imEdge_GPU;
	float *NR_GPU;
	float *G_GPU;
	float *phi_GPU; 
	float *Gx_GPU;
	float *Gy_GPU;
	uint8_t *pedge_GPU; 
	int  size_float = sizeof(float) * width * height;
	int  size_uint8 = sizeof(uint8_t) * width * height;
	cudaMalloc((void **)&im_GPU, size_uint8);
	cudaMalloc((void **)&imEdge_GPU, size_uint8);
	cudaMalloc((void **)&NR_GPU, size_float);
	cudaMalloc((void **)&G_GPU, size_float);
	cudaMalloc((void **)&phi_GPU, size_float);
	cudaMalloc((void **)&Gx_GPU, size_float);
	cudaMalloc((void **)&Gy_GPU, size_float);
	cudaMalloc((void **)&pedge_GPU, size_uint8);

	cudaMemcpy(im_GPU, im, size_uint8, cudaMemcpyHostToDevice);

	dim3 dimBlock(NTHREADS,NTHREADS);
	int blocks_h = (height)/NTHREADS;
	int blocks_w = (width)/NTHREADS;
	if ((height)%NTHREADS>0) blocks_h++;
	if ((width)%NTHREADS>0) blocks_w++;
	dim3 dimGrid(blocks_w, blocks_h);

	gpu_canny_nr<<<dimGrid,dimBlock>>>(im_GPU, imEdge_GPU, NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, pedge_GPU,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();
	
	gpu_canny_gradient<<<dimGrid,dimBlock>>>(im_GPU, imEdge_GPU, NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, pedge_GPU,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();
	gpu_canny_edge<<<dimGrid,dimBlock>>>(im_GPU, imEdge_GPU, NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, pedge_GPU,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();

	dim3 dimBlock_out(NTHREADS, NTHREADS);
	gpu_canny_out<<<dimGrid,dimBlock_out>>>(im_GPU, imEdge_GPU, NR_GPU, G_GPU, phi_GPU, Gx_GPU, Gy_GPU, pedge_GPU,
		1000.0f, //level
		height, width);	
	cudaDeviceSynchronize();

	float* sin_table_GPU;
	float* cos_table_GPU;
	int size_table = 180*sizeof(float);
	cudaMalloc((void **)&sin_table_GPU, size_table);
	cudaMalloc((void **)&cos_table_GPU, size_table);

	cudaMemcpy(sin_table_GPU, sin_table, size_table, cudaMemcpyHostToDevice);
	cudaMemcpy(cos_table_GPU, cos_table, size_table, cudaMemcpyHostToDevice);


	cudaMemcpy(imEdge, imEdge_GPU, size_uint8, cudaMemcpyDeviceToHost);
	cudaMemcpy(NR, NR_GPU, size_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(G, G_GPU, size_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phi, phi_GPU, size_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(Gx, Gx_GPU, size_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(Gy, Gy_GPU, size_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(pedge, pedge_GPU, size_uint8, cudaMemcpyDeviceToHost);

	cudaFree(im_GPU);
	cudaFree(imEdge_GPU);
	cudaFree(NR_GPU);
	cudaFree(G_GPU);
	cudaFree(phi_GPU); 
	cudaFree(Gx_GPU);
	cudaFree(Gy_GPU);
	cudaFree(pedge_GPU); 
	cudaFree(sin_table_GPU);
	cudaFree(cos_table_GPU);

	/* hough transform 
		ejecutamos en CPU porque los accesos a accum pueden tener riesgos de acceso a memoria que producen errores
	*/
	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table);



	if (width>height) threshold = width/6;
	else threshold = height/6;
	
	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}