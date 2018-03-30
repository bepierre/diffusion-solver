#include <iostream>
#include <algorithm>
#include <string.h>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>

#include "timer.hpp"

typedef double value_type;
typedef std::size_t size_type;

#ifndef M_PI
#define M_PI (3.14159265358979)
#endif

class Diffusion2D {
public:
    Diffusion2D(const value_type D,
                const value_type L,
                const size_type N,
                const value_type dt,
		const value_type tmax,
                const size_type M)
            : D_(D), L_(L), N_(N), Ntot_(N_ * N_), M_(M), dt_(dt), tmax_(tmax) {

	/// real space grid spacing
        dr_ = L_ / (N_ - 1);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        rho_.resize(Ntot_, 0.);
        particles_.resize(Ntot_, 0);
	moves_.resize(4*Ntot_, 0.);
	initialize_density();
	initialize_particles();
        }

    void run() {

	VSLStreamStatePtr stream;
	int r[4];
	
	value_type time = 0;
	#pragma omp parallel private(stream) firstprivate(r)
	{
	vslNewStream( &stream, VSL_BRNG_SFMT19937, 123 + omp_get_thread_num());
	while (time < tmax_) {

		#pragma omp for collapse (2)
        	for (size_type i = 1; i < N_ - 1; ++i) {
            	    for (size_type j = 1; j < N_ - 1; ++j) {
	    		if (particles_[i*N_+j] > 0) {
			    do{	
				auto status = viRngBinomial( VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r, particles_[i*N_+j], fac_ );
		      	    } while (r[0] + r[1] + r[2] + r[3] > particles_[i*N_ + j]);
				moves_[(i * 4 * N_) + 4 * j] = r[0];
				moves_[(i * 4 * N_) + 4 * j + 1] = r[1];
				moves_[(i * 4 * N_) + 4 * j + 2] = r[2];
				moves_[(i * 4 * N_) + 4 * j + 3] = r[3];
		        }
            	    }
        	}
		
		#pragma omp single nowait
		for (size_type i = 1; i < N_ - 1; ++i) {
            	    for (size_type j = 1; j < N_ - 1; ++j) {
			particles_[i * N_ + j] -= moves_[(i * 4 * N_) + 4 * j ] + moves_[(i * 4 * N_) + 4 * j + 1] + moves_[(i * 4 * N_) + 4 * j + 2] +moves_[(i * 4 * N_) + 4 * j + 3];
			particles_[(i - 1) * N_ + j] += moves_[(i * 4 * N_) + 4 * j ];
			particles_[(i + 1) * N_ + j] += moves_[(i * 4 * N_) + 4 * j + 1];
			particles_[i * N_ + (j + 1)] += moves_[(i * 4 * N_) + 4 * j + 2];
			particles_[i * N_ + (j - 1)] += moves_[(i * 4 * N_) + 4 * j + 3];
		    }
		}

		#pragma omp single
		{
		time += dt_;
		}
	}
	}
	
	//update rho
	for (size_type i = 1; i < N_ - 1; ++i) {
            	for (size_type j = 1; j < N_ - 1; ++j) {
                	rho_[i * N_ + j] = particles_[i * N_ + j] * 4 / (M_*M_PI*M_PI*dr_*dr_);
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

    void initialize_particles() {
    	size_type num = 0;
        for (size_type i = 1; i < N_-1; ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                particles_[i*N_ + j] = /*particles_tmp_[i*N_+j] =*/ floor(rho_[i*N_ + j] * dr_*dr_*M_PI*M_PI*M_/4);
		num += particles_[i*N_ + j];
            }
        }
	std::cout << num << std::endl;
	//M_ = num;
    }

    value_type D_, L_;
    size_type N_, Ntot_, M_;

    value_type dr_, dt_, fac_, tmax_;

    std::vector <value_type> rho_, rho_tmp;
    std::vector <int> particles_, moves_/*, particles_tmp_*/;
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
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << "option D L N dt T N M" << std::endl;
        return 1;
    } else if (strcmp(argv[1],"-d")==0) {
    
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        const size_type N = std::stoul(argv[6]);
        const size_type M = std::stoul(argv[7]);
	//std::cout << M << std::endl;


        Diffusion2D system(D, L, N, dt, tmax, M);	

        system.write_density("data/rw_000_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

        //const value_type tmax = 10000 * dt;

        timer t;

        t.start();
	
	system.run();
        
        t.stop();

        std::cout << "Timing : " << N << " " << 1 << " " << t.get_timing() << std::endl;
	
	double rms_error = compute_error(system, N, tmax);
	std::cout << "RMS error : " << rms_error << "\n";

        system.write_density("data/rw_serial_N" + std::to_string(N) + "_dt" + std::to_string(dt) + ".dat");

    } else if (strcmp(argv[1],"-c")==0) {

        // Convergence study
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        size_type Nmin = std::stoul(argv[6]);
        size_type Nmax = std::stoul(argv[7]);
        const size_type M = std::stoul(argv[8]);

        std::ofstream out_file("data/conv_serial_" + std::to_string(Nmin) + "_" + std::to_string(Nmax) + ".dat",
                               std::ios::out);

        for (size_type i = Nmin; i <= Nmax; ++i) {
            const size_type N = pow(2, i);

            Diffusion2D system(D, L, N, dt, tmax, M);

            system.run();

            double rms_error = compute_error(system, N, tmax);

            out_file << N << '\t' << rms_error << "\n";

        }
        out_file.close();
    }


    return 0;
}

