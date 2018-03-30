#include <iostream>
#include <algorithm>
#include <string.h>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>

#include "timer.hpp"

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
	
        a_ = c_ = -0.5*fac_;
        b_ = 1 + fac_;
	c_p.resize(N_-3,0.);
	d_p.resize(N_-3,0.);
	c_p[0] = c_/b_;
	
        initialize_density();
        }

    void advance() {

        // First half-step
        std::swap(rho_tmp, rho_);
        for (size_type i = 1; i < N_-1; ++i) {
            
            // First elements
            d_p[0] = (rho_tmp[i*N_+1]+0.5*fac_*(-2*rho_tmp[i*N_+1]+rho_tmp[i*N_+2]))/b_;
        
            // Other elements
            for (size_type j = 1; j < N_-3; ++j) {
                c_p[j] = c_ / (b_ - a_ * c_p[j-1]);
                d_p[j] = ( rho_tmp[i*N_ + 1 + j] + 0.5*fac_*(rho_tmp[i*N_ + j]-2*rho_tmp[i*N_ + 1 + j]+rho_tmp[i*N_ + 2 + j]) - a_*d_p[j-1])  / (b_ - a_ * c_p[j-1]);
            }
            
            // Last element
            rho_[i*N_+(N_-2)] = (rho_tmp[i*N_ + 1 + N_-3] + 0.5*fac_*(rho_tmp[i*N_ + N_-3]-2*rho_tmp[i*N_ + 1 + N_-3]) - a_*d_p[N_-4]) / (b_ - a_ * c_p[N_-4]);
            
            // Backsubstitution
            for (size_type j = N_ - 3; j > 0; --j) {
                rho_[i*N_ + j] = d_p[j-1] - c_p[j-1] * rho_[i*N_+(j+1)];
            }
                      
        }

        // Second half-step
        std::swap(rho_tmp, rho_);
        for(size_type j = 1; j < N_-1; ++j) {
            
            // First elements
            d_p[0] = (rho_tmp[N_+j]+0.5*fac_*(-2*rho_tmp[N_+j]+rho_tmp[2*N_+j]))/b_;
            
            // Other elements
            for (size_type i = 1; i < N_-3; ++i) {
                c_p[i] = c_ / (b_ - a_ * c_p[i-1]);
                d_p[i] = (rho_tmp[(i+1)*N_+j] + 0.5*fac_*(rho_tmp[i*N_+j]-2*rho_tmp[(i+1)*N_+j]+rho_tmp[(i+2)*N_+j])-a_*d_p[i-1])/ (b_ - a_ * c_p[i-1]);
            }
            
            // Last element
            rho_[(N_-2)*N_ + j] = (rho_tmp[(N_-2)*N_ + j] + 0.5*fac_*(rho_tmp[(N_-3)*N_+j]-2*rho_tmp[(N_-2)*N_+j])-a_*d_p[N_-4])/(b_ - a_ * c_p[N_-4]);
            
            // Backsubstitution
            for (size_type i = N_-3; i > 0; --i) {
                rho_[i*N_+j] = d_p[i-1] - c_p[i-1]  * rho_[(i+1)*N_+j];
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

    std::vector <value_type> rho_, rho_tmp, c_p, d_p;
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

        system.write_density("data/density_serial_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

    } else if (strcmp(argv[1],"-c")==0) {

        // Convergence study
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        const size_type Nmin = std::stoul(argv[6]);
        const size_type Nmax = std::stoul(argv[7]);

        std::ofstream out_file("data/conv_serial_" + std::to_string(Nmin) + "_" + std::to_string(Nmax) + ".dat",
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
    } else if (strcmp(argv[1],"-ct")==0) {

        // Convergence study
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const size_type dtmin = std::stod(argv[4]);
	const size_type dtmax = std::stod(argv[5]);
        const size_type N = std::stoul(argv[6]);

        std::ofstream out_file("data/conv_t_serial_" + std::to_string(dtmin) + "_" + std::to_string(dtmax) + ".dat",
                               std::ios::out);
			       
			       
        for (size_type i = dtmin; i <= dtmax; ++i) {
            const value_type dt = 0.000005/pow(2, i);
	    const value_type tmax = dt * 5;

            Diffusion2D system(D, L, N, dt);

            value_type time = 0;
            while (time < tmax) {
                system.advance();
                time += dt;
            }

            double rms_error = compute_error(system, N, tmax);

            out_file << dt << '\t' << rms_error << "\n";

        }
        out_file.close();
    }


    return 0;
}
