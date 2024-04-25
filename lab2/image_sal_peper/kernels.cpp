
#include <CL/sycl.hpp>

using  namespace  cl::sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{

	int ws2 = (window_size-1)>>1; 
    
	Q.submit([&](handler &h) {
        constexpr int tile_size = 16;
        accessor<float, 1, sycl::access::mode::read_write,
        sycl::access::target::local> tile(range<1>(tile_size*tile_size), h);
		h.parallel_for(nd_range<2>{range<2>(height, width), range<2>(tile_size,  tile_size)}, 
        [=](nd_item<2> item){
			float window[9];
			auto n = item.get_global_id()[0];
            auto m = item.get_global_id()[1];
            // Index in the local index space:
            auto j = item.get_local_id()[0];
            auto i = item.get_local_id()[1];
            int ii, jj, kk;
            float tmp;
            float median;
            int size = window_size*window_size;

            tile[i*tile_size+j] = im[m*width+n];
            item.barrier();
            if(m>0 && m<height && n>0 && n<width){
                for (ii =-ws2; ii<=ws2; ii++)
                    for (jj =-ws2; jj<=ws2; jj++)
                        window[(ii+ws2)*window_size + jj+ws2] = tile[(i+1+ii)*width + j+1+jj];

                for (ii=1; ii<size; ii++)
                    for (jj=0 ; jj<size - ii; jj++)
                        if (window[jj] > window[jj+1]){
                            tmp = window[jj];
                            window[jj] = window[jj+1];
                            window[jj+1] = tmp;
                        }

                median = window[(size-1)>>1];

                // Write the final result to global memory.
                if (fabsf((median-im[(m+1)*width+n+1])/median) <=thredshold)
                    image_out[(m+1)*width + n+1] = tile[(i+1)*width+j+1];
                else
                    image_out[(m+1)*width + n+1] = median;
            }
		}); // End of the kernel function
	}).wait(); // End of the queue commands we waint on the event reported.
}
