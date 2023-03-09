
#include <CL/sycl.hpp>

using  namespace  cl::sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{

	int ws2 = (window_size-1)>>1; 
	int H = (height-(ws2*2));
    int W = (width-(ws2*2));
    
	constexpr int tile_size = 16;
	
	Q.submit([&](handler &h) {
		h.parallel_for(range<2>(H, W), [=](id<2> item){
			float window[9];
			auto n = item[0];
			auto m = item[1];
            
            int ii, jj;
            float tmp;
            float median;
            int size = window_size*window_size;

            for (ii =-ws2; ii<=ws2; ii++)
                for (jj =-ws2; jj<=ws2; jj++)
                    window[(ii+ws2)*window_size + jj+ws2] = im[(m+1+ii)*width + n+1+jj];

            for (ii=1; ii<size; ii++)
                for (jj=0 ; jj<size - ii; jj++)
                    if (window[jj] > window[jj+1]){
                        tmp = window[jj];
                        window[jj] = window[jj+1];
                        window[jj+1] = tmp;
                    }

            median = window[(size-1)>>1];
			
			// Write the final result to global memory.
			if (fabsf((median-im[m*width+n])/median) <=thredshold)
				image_out[m*width + n] = im[m*width+n];
			else
				image_out[m*width + n] = median;
		}); // End of the kernel function
	}).wait(); // End of the queue commands we waint on the event reported.
}
