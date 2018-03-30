#include <iostream>
#include <algorithm>
#include <string.h>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>

#include "inc/timer.hpp"

typedef double value_type;
typedef std::size_t size_type;

#ifndef M_PI
#define M_PI (3.1415926535897)
#endif

class Diffusion2D {
public:
    Diffusion2D(const value_type D,
                const value_type L,
                const size_type N,
                const value_type dt,
		const value_type tmax)
            : D_(D), L_(L), N_(N), Ntot_(N_ * N_), dt_(dt), tmax_(tmax) {
        /// real space grid spacing
        dr_ = L_ / (N_ - 1);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        rho_.resize(Ntot_, 0.);
        rho_tmp.resize(Ntot_, 0.);
	
	//Thomas algorithm paramters
	initialize_thomas();
	
	//initialize density
        initialize_density();
        }

    void run() {
    	value_type time = 0;
	
	std::vector<value_type> d_p_;
	d_p_.resize(8*(N_-3),0.);
	
	#pragma omp parallel firstprivate(d_p_)
	{
	while (time < tmax_) {
        // First half-step
	// 8 at a time so that the compiler can vectorize
	#pragma omp single
	{
        std::swap(rho_tmp, rho_);
	}
	
	#pragma omp for nowait
        for (size_type i = 1; i < N_-8; i+=8) {
            
            // First elements
            d_p_[0] = (rho_tmp[i*N_+1]+0.5*fac_*(-2*rho_tmp[i*N_+1]+rho_tmp[i*N_+2]))/b_;
            d_p_[1] = (rho_tmp[(i+1)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+1)*N_+1]+rho_tmp[(i+1)*N_+2]))/b_;
            d_p_[2] = (rho_tmp[(i+2)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+2)*N_+1]+rho_tmp[(i+2)*N_+2]))/b_;
            d_p_[3] = (rho_tmp[(i+3)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+3)*N_+1]+rho_tmp[(i+3)*N_+2]))/b_;
	    d_p_[4] = (rho_tmp[(i+4)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+4)*N_+1]+rho_tmp[(i+4)*N_+2]))/b_;
	    d_p_[5] = (rho_tmp[(i+5)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+5)*N_+1]+rho_tmp[(i+5)*N_+2]))/b_;
	    d_p_[6] = (rho_tmp[(i+6)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+6)*N_+1]+rho_tmp[(i+6)*N_+2]))/b_;
	    d_p_[7] = (rho_tmp[(i+7)*N_+1]+0.5*fac_*(-2*rho_tmp[(i+7)*N_+1]+rho_tmp[(i+7)*N_+2]))/b_;
	    
        
            // Other elements
            for (size_type j = 1; j < N_-3; ++j) {
		d_p_[8*j] = ( rho_tmp[i*N_ + 1 + j] + 0.5*fac_*(rho_tmp[i*N_ + j]-2*rho_tmp[i*N_ + 1 + j]+rho_tmp[i*N_ + 2 + j]) - a_*d_p_[8*(j-1)])  * denom_[j-1];
                d_p_[8*j+1] = ( rho_tmp[(i+1)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+1)*N_ + j]-2*rho_tmp[(i+1)*N_ + 1 + j]+rho_tmp[(i+1)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+1]) * denom_[j-1];
                d_p_[8*j+2] = ( rho_tmp[(i+2)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+2)*N_ + j]-2*rho_tmp[(i+2)*N_ + 1 + j]+rho_tmp[(i+2)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+2]) * denom_[j-1];
                d_p_[8*j+3] = ( rho_tmp[(i+3)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+3)*N_ + j]-2*rho_tmp[(i+3)*N_ + 1 + j]+rho_tmp[(i+3)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+3]) * denom_[j-1];
		d_p_[8*j+4] = ( rho_tmp[(i+4)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+4)*N_ + j]-2*rho_tmp[(i+4)*N_ + 1 + j]+rho_tmp[(i+4)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+4]) * denom_[j-1];
		d_p_[8*j+5] = ( rho_tmp[(i+5)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+5)*N_ + j]-2*rho_tmp[(i+5)*N_ + 1 + j]+rho_tmp[(i+5)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+5]) * denom_[j-1];
		d_p_[8*j+6] = ( rho_tmp[(i+6)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+6)*N_ + j]-2*rho_tmp[(i+6)*N_ + 1 + j]+rho_tmp[(i+6)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+6]) * denom_[j-1];
		d_p_[8*j+7] = ( rho_tmp[(i+7)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+7)*N_ + j]-2*rho_tmp[(i+7)*N_ + 1 + j]+rho_tmp[(i+7)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+7]) * denom_[j-1];
            }
            
            // Last element
            rho_[i*N_+(N_-2)] = (rho_tmp[i*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[i*N_ + N_-3]-2*rho_tmp[i*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-8]) * denom_[N_-5];
	    rho_[(i+1)*N_+(N_-2)] = (rho_tmp[(i+1)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+1)*N_ + N_-3]-2*rho_tmp[(i+1)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-7]) * denom_[N_-5];
	    rho_[(i+2)*N_+(N_-2)] = (rho_tmp[(i+2)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+2)*N_ + N_-3]-2*rho_tmp[(i+2)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-6]) * denom_[N_-5];
	    rho_[(i+3)*N_+(N_-2)] = (rho_tmp[(i+3)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+3)*N_ + N_-3]-2*rho_tmp[(i+3)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-5]) * denom_[N_-5];
	    rho_[(i+4)*N_+(N_-2)] = (rho_tmp[(i+4)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+4)*N_ + N_-3]-2*rho_tmp[(i+4)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-4]) * denom_[N_-5];
	    rho_[(i+5)*N_+(N_-2)] = (rho_tmp[(i+5)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+5)*N_ + N_-3]-2*rho_tmp[(i+5)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-3]) * denom_[N_-5];
	    rho_[(i+6)*N_+(N_-2)] = (rho_tmp[(i+6)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+6)*N_ + N_-3]-2*rho_tmp[(i+6)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-2]) * denom_[N_-5];
	    rho_[(i+7)*N_+(N_-2)] = (rho_tmp[(i+7)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+7)*N_ + N_-3]-2*rho_tmp[(i+7)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-1]) * denom_[N_-5];

	    
            
            // Backsubstitution
	    //std::cout << i << std::endl;
            for (size_type j = N_ - 3; j > 0; --j) {
                rho_[i*N_ + j] = d_p_[8*(j-1)] - c_p_[j-1] * rho_[i*N_+(j+1)];
		rho_[(i+1)*N_ + j] = d_p_[8*(j-1)+1] - c_p_[j-1] * rho_[(i+1)*N_+(j+1)];
		rho_[(i+2)*N_ + j] = d_p_[8*(j-1)+2] - c_p_[j-1] * rho_[(i+2)*N_+(j+1)];
		rho_[(i+3)*N_ + j] = d_p_[8*(j-1)+3] - c_p_[j-1] * rho_[(i+3)*N_+(j+1)];
		rho_[(i+4)*N_ + j] = d_p_[8*(j-1)+4] - c_p_[j-1] * rho_[(i+4)*N_+(j+1)];
		rho_[(i+5)*N_ + j] = d_p_[8*(j-1)+5] - c_p_[j-1] * rho_[(i+5)*N_+(j+1)];
		rho_[(i+6)*N_ + j] = d_p_[8*(j-1)+6] - c_p_[j-1] * rho_[(i+6)*N_+(j+1)];
		rho_[(i+7)*N_ + j] = d_p_[8*(j-1)+7] - c_p_[j-1] * rho_[(i+7)*N_+(j+1)];
            }              
        }
	
	// Extra lines
	#pragma omp single
	for (size_type i=N_-1 - (N_-1)%8; i < N_-1; ++i) {
	    // First elements
            d_p_[0] = (rho_tmp[i*N_+1]+0.5*fac_*(-2*rho_tmp[i*N_+1]+rho_tmp[i*N_+2]))/b_;
        
            // Other elements
            for (size_type j = 1; j < N_-3; ++j) {
                c_p_[j] = c_ / (b_ - a_ * c_p_[j-1]);
                d_p_[j] = ( rho_tmp[i*N_ + 1 + j] + 0.5*fac_*(rho_tmp[i*N_ + j]-2*rho_tmp[i*N_ + 1 + j]+rho_tmp[i*N_ + 2 + j]) - a_*d_p_[j-1]) * denom_[j-1];
            }
            
            // Last element
            rho_[i*N_+(N_-2)] = (rho_tmp[i*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[i*N_ + N_-3]-2*rho_tmp[i*N_ + 1 + N_-3]) - a_*d_p_[N_-4]) * denom_[N_-5];
            
            // Backsubstitution
            for (size_type j = N_ - 3; j > 0; --j) {
                rho_[i*N_ + j] = d_p_[j-1] - c_p_[j-1] * rho_[i*N_+(j+1)];
            } 
	} 	

        // Second half-step
	// 8 at a time so that the compiler can vectorize
	#pragma omp single
	{
        std::swap(rho_tmp, rho_);
	}
	#pragma omp for nowait	
        for(size_type j = 1; j < N_-8; j+=8) {
            
            // First elements
            d_p_[0] = (rho_tmp[N_+j]+0.5*fac_*(-2*rho_tmp[N_+j]+rho_tmp[2*N_+j]))/b_;
            d_p_[1] = (rho_tmp[N_+(j+1)]+0.5*fac_*(-2*rho_tmp[N_+(j+1)]+rho_tmp[2*N_+(j+1)]))/b_;
            d_p_[2] = (rho_tmp[N_+(j+2)]+0.5*fac_*(-2*rho_tmp[N_+(j+2)]+rho_tmp[2*N_+(j+2)]))/b_;
            d_p_[3] = (rho_tmp[N_+(j+3)]+0.5*fac_*(-2*rho_tmp[N_+(j+3)]+rho_tmp[2*N_+(j+3)]))/b_;
	    d_p_[4] = (rho_tmp[N_+(j+4)]+0.5*fac_*(-2*rho_tmp[N_+(j+4)]+rho_tmp[2*N_+(j+4)]))/b_;
	    d_p_[5] = (rho_tmp[N_+(j+5)]+0.5*fac_*(-2*rho_tmp[N_+(j+5)]+rho_tmp[2*N_+(j+5)]))/b_;
	    d_p_[6] = (rho_tmp[N_+(j+6)]+0.5*fac_*(-2*rho_tmp[N_+(j+6)]+rho_tmp[2*N_+(j+6)]))/b_;
	    d_p_[7] = (rho_tmp[N_+(j+7)]+0.5*fac_*(-2*rho_tmp[N_+(j+7)]+rho_tmp[2*N_+(j+7)]))/b_;
            
            // Other elements
            for (size_type i = 1; i < N_-3; ++i) {
                d_p_[8*i] = (rho_tmp[(i+1)*N_+j] + 0.5*fac_*(rho_tmp[i*N_+j]-2*rho_tmp[(i+1)*N_+j]+rho_tmp[(i+2)*N_+j])-a_*d_p_[8*(i-1)]) * denom_[i-1];
                d_p_[8*i+1] = (rho_tmp[(i+1)*N_+(j+1)] + 0.5*fac_*(rho_tmp[i*N_+(j+1)]-2*rho_tmp[(i+1)*N_+(j+1)]+rho_tmp[(i+2)*N_+(j+1)])-a_*d_p_[8*(i-1)+1]) * denom_[i-1];
                d_p_[8*i+2] = (rho_tmp[(i+1)*N_+(j+2)] + 0.5*fac_*(rho_tmp[i*N_+(j+2)]-2*rho_tmp[(i+1)*N_+(j+2)]+rho_tmp[(i+2)*N_+(j+2)])-a_*d_p_[8*(i-1)+2]) * denom_[i-1];
                d_p_[8*i+3] = (rho_tmp[(i+1)*N_+(j+3)] + 0.5*fac_*(rho_tmp[i*N_+(j+3)]-2*rho_tmp[(i+1)*N_+(j+3)]+rho_tmp[(i+2)*N_+(j+3)])-a_*d_p_[8*(i-1)+3]) * denom_[i-1];
		d_p_[8*i+4] = (rho_tmp[(i+1)*N_+(j+4)] + 0.5*fac_*(rho_tmp[i*N_+(j+4)]-2*rho_tmp[(i+1)*N_+(j+4)]+rho_tmp[(i+2)*N_+(j+4)])-a_*d_p_[8*(i-1)+4]) * denom_[i-1];
		d_p_[8*i+5] = (rho_tmp[(i+1)*N_+(j+5)] + 0.5*fac_*(rho_tmp[i*N_+(j+5)]-2*rho_tmp[(i+1)*N_+(j+5)]+rho_tmp[(i+2)*N_+(j+5)])-a_*d_p_[8*(i-1)+5]) * denom_[i-1];
		d_p_[8*i+6] = (rho_tmp[(i+1)*N_+(j+6)] + 0.5*fac_*(rho_tmp[i*N_+(j+6)]-2*rho_tmp[(i+1)*N_+(j+6)]+rho_tmp[(i+2)*N_+(j+6)])-a_*d_p_[8*(i-1)+6]) * denom_[i-1];
		d_p_[8*i+7] = (rho_tmp[(i+1)*N_+(j+7)] + 0.5*fac_*(rho_tmp[i*N_+(j+7)]-2*rho_tmp[(i+1)*N_+(j+7)]+rho_tmp[(i+2)*N_+(j+7)])-a_*d_p_[8*(i-1)+7]) * denom_[i-1];
            }
            
            // Last element
            rho_[(N_-2)*N_ + j] = (rho_tmp[(N_-2)*N_ + j] + 0.5*fac_*(rho_tmp[(N_-3)*N_+j]-2*rho_tmp[(N_-2)*N_+j])-a_*d_p_[8*(N_-3)-8]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+1)] = (rho_tmp[(N_-2)*N_ + (j+1)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+1)]-2*rho_tmp[(N_-2)*N_+(j+1)])-a_*d_p_[8*(N_-3)-7]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+2)] = (rho_tmp[(N_-2)*N_ + (j+2)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+2)]-2*rho_tmp[(N_-2)*N_+(j+2)])-a_*d_p_[8*(N_-3)-6]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+3)] = (rho_tmp[(N_-2)*N_ + (j+3)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+3)]-2*rho_tmp[(N_-2)*N_+(j+3)])-a_*d_p_[8*(N_-3)-5]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+4)] = (rho_tmp[(N_-2)*N_ + (j+4)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+4)]-2*rho_tmp[(N_-2)*N_+(j+4)])-a_*d_p_[8*(N_-3)-4]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+5)] = (rho_tmp[(N_-2)*N_ + (j+5)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+5)]-2*rho_tmp[(N_-2)*N_+(j+5)])-a_*d_p_[8*(N_-3)-3]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+6)] = (rho_tmp[(N_-2)*N_ + (j+6)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+6)]-2*rho_tmp[(N_-2)*N_+(j+6)])-a_*d_p_[8*(N_-3)-2]) * denom_[N_-5];
	    rho_[(N_-2)*N_ + (j+7)] = (rho_tmp[(N_-2)*N_ + (j+7)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+7)]-2*rho_tmp[(N_-2)*N_+(j+7)])-a_*d_p_[8*(N_-3)-1]) * denom_[N_-5];
            
            // Backsubstitution
            for (size_type i = N_-3; i > 0; --i) {
                rho_[i*N_+j] = d_p_[8*(i-1)] - c_p_[i-1]  * rho_[(i+1)*N_+j];
		rho_[i*N_+(j+1)] = d_p_[8*(i-1)+1] - c_p_[i-1]  * rho_[(i+1)*N_+(j+1)];
		rho_[i*N_+(j+2)] = d_p_[8*(i-1)+2] - c_p_[i-1]  * rho_[(i+1)*N_+(j+2)];
		rho_[i*N_+(j+3)] = d_p_[8*(i-1)+3] - c_p_[i-1]  * rho_[(i+1)*N_+(j+3)];
		rho_[i*N_+(j+4)] = d_p_[8*(i-1)+4] - c_p_[i-1]  * rho_[(i+1)*N_+(j+4)];
		rho_[i*N_+(j+5)] = d_p_[8*(i-1)+5] - c_p_[i-1]  * rho_[(i+1)*N_+(j+5)];
		rho_[i*N_+(j+6)] = d_p_[8*(i-1)+6] - c_p_[i-1]  * rho_[(i+1)*N_+(j+6)];
		rho_[i*N_+(j+7)] = d_p_[8*(i-1)+7] - c_p_[i-1]  * rho_[(i+1)*N_+(j+7)];
            }
        }
	
	// Extra columns
	#pragma omp single nowait
	for (size_type j = N_-1 -(N_-1)%8; j < N_-1; ++j) {
	    // First elements
            d_p_[0] = (rho_tmp[N_+j]+0.5*fac_*(-2*rho_tmp[N_+j]+rho_tmp[2*N_+j]))/b_;
            
            // Other elements
            for (size_type i = 1; i < N_-3; ++i) {
                c_p_[i] = c_ / (b_ - a_ * c_p_[i-1]);
                d_p_[i] = (rho_tmp[(i+1)*N_+j] + 0.5*fac_*(rho_tmp[i*N_+j]-2*rho_tmp[(i+1)*N_+j]+rho_tmp[(i+2)*N_+j])-a_*d_p_[i-1]) * denom_[i-1];
            }
            
            // Last element
            rho_[(N_-2)*N_ + j] = (rho_tmp[(N_-2)*N_ + j] + 0.5*fac_*(rho_tmp[(N_-3)*N_+j]-2*rho_tmp[(N_-2)*N_+j])-a_*d_p_[N_-4]) * denom_[N_-5];
            
            // Backsubstitution
            for (size_type i = N_-3; i > 0; --i) {
                rho_[i*N_+j] = d_p_[i-1] - c_p_[i-1]  * rho_[(i+1)*N_+j];
            }
	}
	
	#pragma omp single
	{
	time +=dt_;
	}
	
	}
    }  
    }

    void write_density(std::string const &filename) const {
        std::ofstream out_file(filename, std::ios::out);

        for (size_type i = 0; i < N_; ++i) {
            for (size_type j = 0; j < N_; ++j)
                out_file << (i * dr_ - L_ / 2.) << '\t' << (j * dr_ - L_ / 2.) << '\t' << rho_[i * N_ + j] << "\n";
            out_file << "\n";
        }
        out_file.close();
    }
    
    double exact_rho(size_type i, size_type j, double t) {
        return sin(M_PI*i*dr_)*sin(M_PI*j*dr_)*exp(-2*D_*M_PI*M_PI*t);
    }
    
    std::vector<value_type> get_rho() {
        return rho_;
    }

private:

    void initialize_density() {
        /// initialize rho(x,y,t=0)
        for (size_type i = 1; i < N_-1; ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                rho_[i * N_ + j] = sin(M_PI * i * dr_) * sin(M_PI * j * dr_);
            }
        }
    }
    
    void initialize_thomas() {
        a_ = c_ = -0.5*fac_;
        b_ = 1 + fac_;
	c_p_.resize((N_-3), 0.);
	denom_.resize((N_-4), 0.);
	c_p_[0] = c_/b_;
	for (size_type i = 1; i < N_-3; ++i) {
		denom_[i-1] = 1 / (b_ - a_ * c_p_[i-1]);
                c_p_[i] = c_ * denom_[i-1];
	}
    }

    value_type D_, L_;
    size_type N_, Ntot_;

    value_type dr_, dt_, fac_, a_, b_, c_, tmax_;

    std::vector <value_type> rho_, rho_tmp, c_p_, denom_;
};



double compute_error(Diffusion2D system, size_type N, double T) {
    
    std::vector<value_type> rho_h = system.get_rho();
    double rms = 0;
    for (value_type i = 0; i < N; ++i) {
        for (value_type j = 0; j < N; ++j) {
            rms += pow((rho_h[i*N+j] - system.exact_rho(i,j,T)),2);
        }
    }
    return sqrt(rms/(N*N));
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << "option D L N dt T N" << std::endl;
        return 1;
    } else if (strcmp(argv[1],"-d")==0) {

        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        const size_type N = std::stoul(argv[6]);


        Diffusion2D system(D, L, N, dt, tmax);
        system.write_density("data/density_000_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

        value_type time = 0;

        timer t;

        t.start();
	
        system.run();
	
        t.stop();

        std::cout << "Timing : " << N << " " << 1 << " " << t.get_timing() << std::endl;

        system.write_density("data/density_unroll_8_mt_run_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

    } else if (strcmp(argv[1],"-c")==0) {

        // Convergence study
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        size_type Nmin = std::stoul(argv[6]);
        size_type Nmax = std::stoul(argv[7]);

        std::ofstream out_file("data/conv_unroll_8_mt_run_" + std::to_string(Nmin) + "_" + std::to_string(Nmax) + ".dat",
                               std::ios::out);

        for (size_type i = Nmin; i <= Nmax; ++i) {
            const size_type N = pow(2, i);

            Diffusion2D system(D, L, N, dt, tmax);

            system.run();

            double rms_error = compute_error(system, N, tmax);

            out_file << N << '\t' << rms_error << "\n";

        }
        out_file.close();
    }


    return 0;
}
