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
                const value_type dt)
            : D_(D), L_(L), N_(N), Ntot_(N_ * N_), dt_(dt) {
        /// real space grid spacing
        dr_ = L_ / (N_ - 1);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        rho_.resize(Ntot_, 0.);
        rho_tmp.resize(Ntot_, 0.);
	
	//Thomas algorithm paramters
        a_ = c_ = -0.5*fac_;
        b_ = 1 + fac_;
	c_p_.resize(8*(N_-3), 0.);
        d_p_.resize(8*(N_-3), 0.);
	c_p_[0] = c_p_[1] = c_p_[2] = c_p_[3] = c_p_[4] = c_p_[5] = c_p_[6] = c_p_[7] = c_/b_;
	
	//initialize density
        initialize_density();
        }

    void advance() {
        // First half-step
	// 8 at a time so that the compiler can vectorize
        std::swap(rho_tmp, rho_);
	c_p_[0] = c_p_[1] = c_p_[2] = c_p_[3] = c_p_[4] = c_p_[5] = c_p_[6] = c_p_[7] = c_/b_;
	
	size_type i;
        for (i = 1; i < N_-8; i+=8) {
            
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
                c_p_[8*j] = c_ / (b_ - a_ * c_p_[8*(j-1)]);
		c_p_[8*j+1] = c_ / (b_ - a_ * c_p_[8*(j-1)+1]);
		c_p_[8*j+2] = c_ / (b_ - a_ * c_p_[8*(j-1)+2]);
		c_p_[8*j+3] = c_ / (b_ - a_ * c_p_[8*(j-1)+3]);
		c_p_[8*j+4] = c_ / (b_ - a_ * c_p_[8*(j-1)+4]);
		c_p_[8*j+5] = c_ / (b_ - a_ * c_p_[8*(j-1)+5]);
		c_p_[8*j+6] = c_ / (b_ - a_ * c_p_[8*(j-1)+6]);
		c_p_[8*j+7] = c_ / (b_ - a_ * c_p_[8*(j-1)+7]);
                
		d_p_[8*j] = ( rho_tmp[i*N_ + 1 + j] + 0.5*fac_*(rho_tmp[i*N_ + j]-2*rho_tmp[i*N_ + 1 + j]+rho_tmp[i*N_ + 2 + j]) - a_*d_p_[8*(j-1)])  / (b_ - a_ * c_p_[8*(j-1)]);
                d_p_[8*j+1] = ( rho_tmp[(i+1)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+1)*N_ + j]-2*rho_tmp[(i+1)*N_ + 1 + j]+rho_tmp[(i+1)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+1])  / (b_ - a_ * c_p_[8*(j-1)+1]);
                d_p_[8*j+2] = ( rho_tmp[(i+2)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+2)*N_ + j]-2*rho_tmp[(i+2)*N_ + 1 + j]+rho_tmp[(i+2)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+2])  / (b_ - a_ * c_p_[8*(j-1)+2]);
                d_p_[8*j+3] = ( rho_tmp[(i+3)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+3)*N_ + j]-2*rho_tmp[(i+3)*N_ + 1 + j]+rho_tmp[(i+3)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+3])  / (b_ - a_ * c_p_[8*(j-1)+3]);
		d_p_[8*j+4] = ( rho_tmp[(i+4)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+4)*N_ + j]-2*rho_tmp[(i+4)*N_ + 1 + j]+rho_tmp[(i+4)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+4])  / (b_ - a_ * c_p_[8*(j-1)+4]);
		d_p_[8*j+5] = ( rho_tmp[(i+5)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+5)*N_ + j]-2*rho_tmp[(i+5)*N_ + 1 + j]+rho_tmp[(i+5)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+5])  / (b_ - a_ * c_p_[8*(j-1)+5]);
		d_p_[8*j+6] = ( rho_tmp[(i+6)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+6)*N_ + j]-2*rho_tmp[(i+6)*N_ + 1 + j]+rho_tmp[(i+6)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+6])  / (b_ - a_ * c_p_[8*(j-1)+6]);
		d_p_[8*j+7] = ( rho_tmp[(i+7)*N_ + 1 + j] + 0.5*fac_*(rho_tmp[(i+7)*N_ + j]-2*rho_tmp[(i+7)*N_ + 1 + j]+rho_tmp[(i+7)*N_ + 2 + j]) - a_*d_p_[8*(j-1)+7])  / (b_ - a_ * c_p_[8*(j-1)+7]);
            }
            
            // Last element
            rho_[i*N_+(N_-2)] = (rho_tmp[i*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[i*N_ + N_-3]-2*rho_tmp[i*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-8]) / (b_ - a_ * c_p_[8*(N_-3)-8]);
	    rho_[(i+1)*N_+(N_-2)] = (rho_tmp[(i+1)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+1)*N_ + N_-3]-2*rho_tmp[(i+1)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-7]) / (b_ - a_ * c_p_[8*(N_-3)-7]);
	    rho_[(i+2)*N_+(N_-2)] = (rho_tmp[(i+2)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+2)*N_ + N_-3]-2*rho_tmp[(i+2)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-6]) / (b_ - a_ * c_p_[8*(N_-3)-6]);
	    rho_[(i+3)*N_+(N_-2)] = (rho_tmp[(i+3)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+3)*N_ + N_-3]-2*rho_tmp[(i+3)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-5]) / (b_ - a_ * c_p_[8*(N_-3)-5]);
	    rho_[(i+4)*N_+(N_-2)] = (rho_tmp[(i+4)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+4)*N_ + N_-3]-2*rho_tmp[(i+4)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-4]) / (b_ - a_ * c_p_[8*(N_-3)-4]);
	    rho_[(i+5)*N_+(N_-2)] = (rho_tmp[(i+5)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+5)*N_ + N_-3]-2*rho_tmp[(i+5)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-3]) / (b_ - a_ * c_p_[8*(N_-3)-3]);
	    rho_[(i+6)*N_+(N_-2)] = (rho_tmp[(i+6)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+6)*N_ + N_-3]-2*rho_tmp[(i+6)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-2]) / (b_ - a_ * c_p_[8*(N_-3)-2]);
	    rho_[(i+7)*N_+(N_-2)] = (rho_tmp[(i+7)*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[(i+7)*N_ + N_-3]-2*rho_tmp[(i+7)*N_ + 1 + N_-3]) - a_*d_p_[8*(N_-3)-1]) / (b_ - a_ * c_p_[8*(N_-3)-1]);

	    
            
            // Backsubstitution
	    //std::cout << i << std::endl;
            for (size_type j = N_ - 3; j > 0; --j) {
                rho_[i*N_ + j] = d_p_[8*(j-1)] - c_p_[8*(j-1)] * rho_[i*N_+(j+1)];
		rho_[(i+1)*N_ + j] = d_p_[8*(j-1)+1] - c_p_[8*(j-1)+1] * rho_[(i+1)*N_+(j+1)];
		rho_[(i+2)*N_ + j] = d_p_[8*(j-1)+2] - c_p_[8*(j-1)+2] * rho_[(i+2)*N_+(j+1)];
		rho_[(i+3)*N_ + j] = d_p_[8*(j-1)+3] - c_p_[8*(j-1)+3] * rho_[(i+3)*N_+(j+1)];
		rho_[(i+4)*N_ + j] = d_p_[8*(j-1)+4] - c_p_[8*(j-1)+4] * rho_[(i+4)*N_+(j+1)];
		rho_[(i+5)*N_ + j] = d_p_[8*(j-1)+5] - c_p_[8*(j-1)+5] * rho_[(i+5)*N_+(j+1)];
		rho_[(i+6)*N_ + j] = d_p_[8*(j-1)+6] - c_p_[8*(j-1)+6] * rho_[(i+6)*N_+(j+1)];
		rho_[(i+7)*N_ + j] = d_p_[8*(j-1)+7] - c_p_[8*(j-1)+7] * rho_[(i+7)*N_+(j+1)];
            }              
        }
	
	// Extra lines
	for (size_type k = i; k < N_-1; ++k) {
	    // First elements
	    d_p_[0] = (rho_tmp[k*N_+1]+0.5*fac_*(-2*rho_tmp[k*N_+1]+rho_tmp[k*N_+2]))/b_;
	    
	    // Other elements
            for (size_type j = 1; j < N_-3; ++j) {
                c_p_[j] = c_ / (b_ - a_ * c_p_[j-1]);
                d_p_[j] = ( rho_tmp[k*N_ + 1 + j] + 0.5*fac_*(rho_tmp[k*N_ + j]-2*rho_tmp[k*N_ + 1 + j]+rho_tmp[k*N_ + 2 + j]) - a_*d_p_[j-1])  / (b_ - a_ * c_p_[j-1]);
            }
            
            // Last element
            rho_[k*N_+(N_-2)] = (rho_tmp[k*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[k*N_ + N_-3]-2*rho_tmp[k*N_ + 1 + N_-3]) - a_*d_p_[N_-4]) / (b_ - a_ * c_p_[N_-4]);
            
            // Backsubstitution
            for (size_type j = N_ - 3; j > 0; --j) {
                rho_[k*N_ + j] = d_p_[j-1] - c_p_[j-1] * rho_[k*N_+(j+1)];
            } 
	} 	

        // Second half-step
	// 8 at a time so that the compiler can vectorize
        std::swap(rho_tmp, rho_);
	c_p_[0] = c_p_[1] = c_p_[2] = c_p_[3] = c_p_[4] = c_p_[5] = c_p_[6] = c_p_[7] = c_/b_;
	
	size_type j;
        for(j = 1; j < N_-8; j+=8) {
            
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
                c_p_[8*i] = c_ / (b_ - a_ * c_p_[8*(i-1)]);
		c_p_[8*i+1] = c_ / (b_ - a_ * c_p_[8*(i-1)+1]);
		c_p_[8*i+2] = c_ / (b_ - a_ * c_p_[8*(i-1)+2]);
		c_p_[8*i+3] = c_ / (b_ - a_ * c_p_[8*(i-1)+3]);
		c_p_[8*i+4] = c_ / (b_ - a_ * c_p_[8*(i-1)+4]);
		c_p_[8*i+5] = c_ / (b_ - a_ * c_p_[8*(i-1)+5]);
		c_p_[8*i+6] = c_ / (b_ - a_ * c_p_[8*(i-1)+6]);
		c_p_[8*i+7] = c_ / (b_ - a_ * c_p_[8*(i-1)+7]);

                d_p_[8*i] = (rho_tmp[(i+1)*N_+j] + 0.5*fac_*(rho_tmp[i*N_+j]-2*rho_tmp[(i+1)*N_+j]+rho_tmp[(i+2)*N_+j])-a_*d_p_[8*(i-1)])/ (b_ - a_ * c_p_[8*(i-1)]);
                d_p_[8*i+1] = (rho_tmp[(i+1)*N_+(j+1)] + 0.5*fac_*(rho_tmp[i*N_+(j+1)]-2*rho_tmp[(i+1)*N_+(j+1)]+rho_tmp[(i+2)*N_+(j+1)])-a_*d_p_[8*(i-1)+1])/ (b_ - a_ * c_p_[8*(i-1)+1]);
                d_p_[8*i+2] = (rho_tmp[(i+1)*N_+(j+2)] + 0.5*fac_*(rho_tmp[i*N_+(j+2)]-2*rho_tmp[(i+1)*N_+(j+2)]+rho_tmp[(i+2)*N_+(j+2)])-a_*d_p_[8*(i-1)+2])/ (b_ - a_ * c_p_[8*(i-1)+2]);
                d_p_[8*i+3] = (rho_tmp[(i+1)*N_+(j+3)] + 0.5*fac_*(rho_tmp[i*N_+(j+3)]-2*rho_tmp[(i+1)*N_+(j+3)]+rho_tmp[(i+2)*N_+(j+3)])-a_*d_p_[8*(i-1)+3])/ (b_ - a_ * c_p_[8*(i-1)+3]);
		d_p_[8*i+4] = (rho_tmp[(i+1)*N_+(j+4)] + 0.5*fac_*(rho_tmp[i*N_+(j+4)]-2*rho_tmp[(i+1)*N_+(j+4)]+rho_tmp[(i+2)*N_+(j+4)])-a_*d_p_[8*(i-1)+4])/ (b_ - a_ * c_p_[8*(i-1)+4]);
		d_p_[8*i+5] = (rho_tmp[(i+1)*N_+(j+5)] + 0.5*fac_*(rho_tmp[i*N_+(j+5)]-2*rho_tmp[(i+1)*N_+(j+5)]+rho_tmp[(i+2)*N_+(j+5)])-a_*d_p_[8*(i-1)+5])/ (b_ - a_ * c_p_[8*(i-1)+5]);
		d_p_[8*i+6] = (rho_tmp[(i+1)*N_+(j+6)] + 0.5*fac_*(rho_tmp[i*N_+(j+6)]-2*rho_tmp[(i+1)*N_+(j+6)]+rho_tmp[(i+2)*N_+(j+6)])-a_*d_p_[8*(i-1)+6])/ (b_ - a_ * c_p_[8*(i-1)+6]);
		d_p_[8*i+7] = (rho_tmp[(i+1)*N_+(j+7)] + 0.5*fac_*(rho_tmp[i*N_+(j+7)]-2*rho_tmp[(i+1)*N_+(j+7)]+rho_tmp[(i+2)*N_+(j+7)])-a_*d_p_[8*(i-1)+7])/ (b_ - a_ * c_p_[8*(i-1)+7]);
            }
            
            // Last element
            rho_[(N_-2)*N_ + j] = (rho_tmp[(N_-2)*N_ + j] + 0.5*fac_*(rho_tmp[(N_-3)*N_+j]-2*rho_tmp[(N_-2)*N_+j])-a_*d_p_[8*(N_-3)-8])/(b_ - a_ * c_p_[8*(N_-3)-8]);
	    rho_[(N_-2)*N_ + (j+1)] = (rho_tmp[(N_-2)*N_ + (j+1)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+1)]-2*rho_tmp[(N_-2)*N_+(j+1)])-a_*d_p_[8*(N_-3)-7])/(b_ - a_ * c_p_[8*(N_-3)-7]);
	    rho_[(N_-2)*N_ + (j+2)] = (rho_tmp[(N_-2)*N_ + (j+2)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+2)]-2*rho_tmp[(N_-2)*N_+(j+2)])-a_*d_p_[8*(N_-3)-6])/(b_ - a_ * c_p_[8*(N_-3)-6]);
	    rho_[(N_-2)*N_ + (j+3)] = (rho_tmp[(N_-2)*N_ + (j+3)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+3)]-2*rho_tmp[(N_-2)*N_+(j+3)])-a_*d_p_[8*(N_-3)-5])/(b_ - a_ * c_p_[8*(N_-3)-5]);
	    rho_[(N_-2)*N_ + (j+4)] = (rho_tmp[(N_-2)*N_ + (j+4)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+4)]-2*rho_tmp[(N_-2)*N_+(j+4)])-a_*d_p_[8*(N_-3)-4])/(b_ - a_ * c_p_[8*(N_-3)-4]);
	    rho_[(N_-2)*N_ + (j+5)] = (rho_tmp[(N_-2)*N_ + (j+5)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+5)]-2*rho_tmp[(N_-2)*N_+(j+5)])-a_*d_p_[8*(N_-3)-3])/(b_ - a_ * c_p_[8*(N_-3)-3]);
	    rho_[(N_-2)*N_ + (j+6)] = (rho_tmp[(N_-2)*N_ + (j+6)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+6)]-2*rho_tmp[(N_-2)*N_+(j+6)])-a_*d_p_[8*(N_-3)-2])/(b_ - a_ * c_p_[8*(N_-3)-2]);
	    rho_[(N_-2)*N_ + (j+7)] = (rho_tmp[(N_-2)*N_ + (j+7)] + 0.5*fac_*(rho_tmp[(N_-3)*N_+(j+7)]-2*rho_tmp[(N_-2)*N_+(j+7)])-a_*d_p_[8*(N_-3)-1])/(b_ - a_ * c_p_[8*(N_-3)-1]);
            
            // Backsubstitution
            for (size_type i = N_-3; i > 0; --i) {
                rho_[i*N_+j] = d_p_[8*(i-1)] - c_p_[4*(i-1)]  * rho_[(i+1)*N_+j];
		rho_[i*N_+(j+1)] = d_p_[8*(i-1)+1] - c_p_[8*(i-1)+1]  * rho_[(i+1)*N_+(j+1)];
		rho_[i*N_+(j+2)] = d_p_[8*(i-1)+2] - c_p_[8*(i-1)+2]  * rho_[(i+1)*N_+(j+2)];
		rho_[i*N_+(j+3)] = d_p_[8*(i-1)+3] - c_p_[8*(i-1)+3]  * rho_[(i+1)*N_+(j+3)];
		rho_[i*N_+(j+4)] = d_p_[8*(i-1)+4] - c_p_[8*(i-1)+4]  * rho_[(i+1)*N_+(j+4)];
		rho_[i*N_+(j+5)] = d_p_[8*(i-1)+5] - c_p_[8*(i-1)+5]  * rho_[(i+1)*N_+(j+5)];
		rho_[i*N_+(j+6)] = d_p_[8*(i-1)+6] - c_p_[8*(i-1)+6]  * rho_[(i+1)*N_+(j+6)];
		rho_[i*N_+(j+7)] = d_p_[8*(i-1)+7] - c_p_[8*(i-1)+7]  * rho_[(i+1)*N_+(j+7)];
            }
        }
	
	// Extra columns
	for (size_type k = j; k < N_-1; ++k) {
	    // First elements
            d_p_[0] = (rho_tmp[N_+k]+0.5*fac_*(-2*rho_tmp[N_+k]+rho_tmp[(1+1)*N_+k]))/b_;
            
            // Other elements
            for (size_type i = 1; i < N_-3; ++i) {
                c_p_[i] = c_ / (b_ - a_ * c_p_[i-1]);
                d_p_[i] = (rho_tmp[(i+1)*N_+k] + 0.5*fac_*(rho_tmp[i*N_+k]-2*rho_tmp[(i+1)*N_+k]+rho_tmp[(i+2)*N_+k])-a_*d_p_[i-1])/ (b_ - a_ * c_p_[i-1]);
            }
            
            // Last element
            rho_[(N_-2)*N_ + k] = (rho_tmp[(N_-2)*N_ + k] + 0.5*fac_*(rho_tmp[(N_-3)*N_+k]-2*rho_tmp[(N_-2)*N_+k])-a_*d_p_[N_-4])/(b_ - a_ * c_p_[N_-4]);
            
            // Backsubstitution
            for (size_type i = N_-3; i > 0; --i) {
                rho_[i*N_+k] = d_p_[i-1] - c_p_[i-1]  * rho_[(i+1)*N_+k];
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

    value_type D_, L_;
    size_type N_, Ntot_;

    value_type dr_, dt_, fac_, a_, b_, c_;

    std::vector <value_type> rho_, rho_tmp, A, c_p_, d_p_;
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


        Diffusion2D system(D, L, N, dt);
        system.write_density("data/density_000_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

        //const value_type tmax = 10000 * dt;
        value_type time = 0;

        timer t;

        t.start();
        while (time < tmax) {
            system.advance();
            time += dt;
        }
        t.stop();

        std::cout << "Timing : " << N << " " << 1 << " " << t.get_timing() << std::endl;

        system.write_density("data/density_unroll_8_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

    } else if (strcmp(argv[1],"-c")==0) {

        // Convergence study
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        size_type Nmin = std::stoul(argv[6]);
        size_type Nmax = std::stoul(argv[7]);

        std::ofstream out_file("data/conv_unroll_8_" + std::to_string(Nmin) + "_" + std::to_string(Nmax) + ".dat",
                               std::ios::out);

        for (size_type i = Nmin; i <= Nmax; ++i) {
            const size_type N = pow(2, i);

            Diffusion2D system(D, L, N, dt);

            value_type time = 0;
            while (time < tmax) {
                system.advance();
                time += dt;
            }

            double rms_error = compute_error(system, N, tmax);

            out_file << N << '\t' << rms_error << "\n";

        }
        out_file.close();
    }


    return 0;
}
