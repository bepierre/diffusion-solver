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
#include <mpi.h>
#include <mkl.h>

#include "timer.hpp"

typedef double value_type;
#define MPI_VALUE_TYPE MPI_DOUBLE
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
                const size_type M,
		const size_type rank,
		const size_type procs)
            : D_(D), L_(L), N_(N), M_(M), dt_(dt), tmax_(tmax), rank_(rank), procs_(procs) {
       
	/// real space grid spacing
        dr_ = L_ / (N_ - 1);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);
	
	// number of rows per process
	local_N_ = N_ / procs_;
	
	// correction for last process
	if (rank_ == procs - 1 ) {
	    local_N_ += (N_ % procs_);
	}
	
	Ntot_ = (local_N_ + 2) * N_;

        rho_.resize(local_N_*N_, 0.);
        particles_.resize(Ntot_, 0.);
	particles_tmp_.resize(Ntot_, 0.);
        initialize_density();
	initialize_particles();
        }

    void run() {
	
	size_type up, down, left, right;
	
	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_SFMT19937, 777+rank_ );
	int r[4];
	
	value_type time = 0;
        
	while (time < tmax_) {
	
	int prev_rank = rank_ - 1;
	int next_rank = rank_ + 1;

        for (size_type i = 1 + (rank_ == 0) ; i < local_N_ + 1 - (rank_ == procs_ - 1); ++i) {
            for (size_type j = 1; j < N_ - 1; ++j) {
		if (particles_[i*N_+j] > 0) {
			//size_type iter = -1;
			do{	
				auto status = viRngBinomial( VSL_RNG_METHOD_BINOMIAL_BTPE, stream, 4, r, particles_[i*N_+j], fac_ );
		
				up = r[0];
				down = r[1];
				left = r[2];
				right = r[3];
				/*
				if ( ++iter > 1) {
					std::cout << iter << " too many particles want to move \n";
				}
				*/
		      	} while (up + down + left + right > particles_[i *N_ + j]);
		
		    	particles_tmp_[i * N_ + j] -= (up + down + left + right);
		    	particles_tmp_[(i - 1) * N_ + j] += up;
	       	    	particles_tmp_[(i + 1) * N_ + j] += down;
	       	    	particles_tmp_[i * N_ + (j - 1)] += left;
		    	particles_tmp_[i * N_ + (j + 1)] += right;
		}
            }
        }
	
	
	// Swaping ghost lines with other processes
	MPI_Request req[4];
	MPI_Status status[4];

	if (prev_rank >= 0) {
		MPI_Irecv(&particles_tmp_[ 0], N_, MPI_VALUE_TYPE, prev_rank, 100, MPI_COMM_WORLD, &req[0]);
		MPI_Isend(&particles_tmp_[ 0], N_, MPI_VALUE_TYPE, prev_rank, 100, MPI_COMM_WORLD, &req[1]);
	} else {
		req[0] = MPI_REQUEST_NULL;
		req[1] = MPI_REQUEST_NULL;
	}
	
	if (next_rank < procs_) {
		MPI_Irecv(&particles_tmp_[(local_N_+1)*N_], N_, MPI_VALUE_TYPE, next_rank, 100, MPI_COMM_WORLD, &req[2]);
		MPI_Isend(&particles_tmp_[(local_N_+1)*N_], N_, MPI_VALUE_TYPE, next_rank, 100, MPI_COMM_WORLD, &req[3]);
	} else {
		req[2] = MPI_REQUEST_NULL;
		req[3] = MPI_REQUEST_NULL;
	}
	
	MPI_Waitall(4, req, status);
	
	//adding ghost line onto boundary lines
	if (prev_rank >= 0) {
            for (size_type j = 1; j < N_ - 1; ++j) {
                particles_tmp_[N_+j] += particles_tmp_[j];
                particles_tmp_[j] = 0;
            }
	}
	
	if (next_rank < procs_) {
	    for (size_type j = 1; j < N_ - 1; ++j) {
		particles_tmp_[local_N_*N_+j] += particles_tmp_[(local_N_+1)*N_+j];	
                particles_tmp_[(local_N_+1)*N_+j] = 0;
	    }
	}
	
	particles_ = particles_tmp_;
	
	time += dt_;
	}
	
	//update rho
        for (size_type i = (rank_==0); i < local_N_ - (rank_==procs_-1); ++i) {
            for (size_type j = 1; j < N_ - 1; ++j) {
	        rho_[i * N_ + j] = particles_[(i + 1) * N_ + j] * 4 / (M_*M_PI*M_PI*dr_*dr_);
            }
        }
	
    }

    void write_density(std::string const &filename) const {
        std::ofstream out_file(filename, std::ios::out);
	
	size_type line = N_/procs_*rank_;
        for (size_type i = 0; i < local_N_; ++i) {
            for (size_type j = 0; j < N_; ++j)
                out_file << ((i+line) * dr_ - L_ / 2.) << '\t' << (j * dr_ - L_ / 2.) << '\t' << rho_[i * N_ + j] << "\n";
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
    
    double compute_error(double T) {
	
    	double rms = 0; 
    	size_type line = N_/procs_*rank_;
    	/// initialize rho(x,y,t=0)
    	for (size_type i =  0; i < local_N_; ++i) {
    		for (size_type j = 0; j < N_; ++j) {
			//std::cout << "i:" << i+line << " j:" << j << " | rho :" << rho_[i*N_+j] << " | ex :"<< exact_rho(line+i,j,T) << '\n';
         	       rms += pow((rho_[i*N_+j] - exact_rho(line+i,j,T)),2);
		}
    	}
	return (rms/(N_*N_));
    }

private:

    void initialize_density() {
    	size_type line = N_/procs_*rank_;
    
        /// initialize rho(x,y,t=0)
        for (size_type i = (rank_==0); i < local_N_ - (rank_==procs_-1); ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                rho_[i * N_ + j] = sin(M_PI * (line+i) * dr_) * sin(M_PI * j * dr_);
            }
        }
    }

    void initialize_particles() {
    	size_type num = 0;
        for (size_type i = 1; i < local_N_+1; ++i) {
            for (size_type j = 1; j < N_-1; ++j) {
                particles_[i*N_ + j] = particles_tmp_[i*N_+j] = floor(rho_[(i-1)*N_ + j] * dr_*dr_*M_PI*M_PI*M_/4);
		num += particles_[i*N_ + j];
            }
        }
	std::cout << "Rank: " << rank_ << "| local_N_: " << local_N_ << "| num: " << num << '\n';
	//M_ = num;  
    }

    value_type D_, L_;
    size_type N_, Ntot_, local_N_, M_, rank_, procs_;

    value_type dr_, dt_, fac_, tmax_;

    std::vector <value_type> rho_;
    std::vector <size_type> particles_, particles_tmp_;
};

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << "option D L N dt T N M" << std::endl;
        return 1;
    } else if (strcmp(argv[1],"-d")==0) {
    
    	MPI_Init(&argc, &argv);
	
	int rank, procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	
	if (rank == 0) {
	    std::cout << "Running with " << procs << " MPI processes" << std::endl;	
	}
    
        const value_type D = std::stod(argv[2]);
        const value_type L = std::stod(argv[3]);
        const value_type dt = std::stod(argv[4]);
        const value_type tmax = std::stod(argv[5]);
        const size_type N = std::stoul(argv[6]);
        const size_type M = std::stoul(argv[7]);
	//std::cout << M << std::endl;


        Diffusion2D system(D, L, N, dt, tmax, M, rank, procs);
        //system.write_density("new/data/rw_000_N" + std::to_string(N) + "_" + std::to_string(rank) + "_dt" + std::to_string(dt) + ".dat");

        //const value_type tmax = 10000 * dt;
	
	value_type init_error = system.compute_error(0);
	std::cout << "Rank: " << rank << " | Init error : " << init_error << '\n';

        timer t;

        t.start();
	
	system.run();
        
        t.stop();

        std::cout << "Timing : " << N << " " << 1 << " " << t.get_timing() << std::endl;
	
	double rms_error = system.compute_error(tmax);
	std::cout << "Rank: " << rank << " | MS error : " << rms_error << '\n';
	
	//if ( rank == 0) {
        //system.write_density("new/data/rw_mklmpi_N" + std::to_string(N) +"_" + std::to_string(rank) + "_dt" + std::to_string(dt) + ".dat");
	//}
	MPI_Finalize();

    } /*else if (strcmp(argv[1],"-c")==0) {
    
    	MPI_Init(&argc, &argv);
	
	int rank, procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

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
	
	MPI_Finalize();
    }*/


    return 0;
}

