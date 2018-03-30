ifeq ($(USER),stud18)
	CXX=CC
else
	CXX?=g++
endif
CFLAGS=-Wall -std=c++11 -O3
CFLAGS_THREADS=$(CFLAGS) -pthread -fopenmp
CMPI?=mpicxx

all: diffusion2d_serial  diffusion2d_unroll_8_optim  diffusion2d_unroll_8_mt_run diffusion2d_rw_serial diffusion2d_rw_serial_run diffusion2d_rw_mkl diffusion2d_rw_mkl_run diffusion2d_AVX diffusion2d_AVX_u8  diffusion2d_AVX_align diffusion2d_AVX_mt diffusion2d_AVX_u8_mt diffusion2d_rw_mkl_mt diffusion2d_rw_mkl_mpi diffusion2d_rw_mpi diffusion2d_rw_mkl_mt_mpi 
#diffusion2d_MPI

diffusion2d_serial: diffusion2d_serial.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -o $@ $< -Iinc
	
diffusion2d_unroll_8_optim: diffusion2d_unroll_8_optim.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -fno-tree-vectorize -o $@ $< -Iinc

diffusion2d_unroll_8_mt_run: diffusion2d_unroll_8_mt_run.cpp inc/timer.hpp
	$(CXX) $(CFLAGS_THREADS) -o $@ $< -Iinc
	
diffusion2d_AVX: diffusion2d_AVX.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -march=native -o $@ $< -Iinc

diffusion2d_AVX_u8: diffusion2d_AVX_u8.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -march=native -o $@ $< -Iinc

diffusion2d_AVX_align: diffusion2d_AVX_align.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -march=native -o $@ $< -Iinc
	
diffusion2d_AVX_mt: diffusion2d_AVX_mt.cpp inc/timer.hpp
	$(CXX) $(CFLAGS_THREADS) -march=native -o $@ $< -Iinc

diffusion2d_AVX_u8_mt: diffusion2d_AVX_u8_mt.cpp inc/timer.hpp
	$(CXX) $(CFLAGS_THREADS) -march=native -o $@ $< -Iinc

diffusion2d_rw_serial: diffusion2d_rw_serial.cpp inc/timer.hpp
	CC $(CFLAGS) -o $@ $< -Iinc

diffusion2d_rw_serial_run: diffusion2d_rw_serial_run.cpp inc/timer.hpp
	CC $(CFLAGS) -o $@ $< -Iinc

diffusion2d_rw_mkl: diffusion2d_rw_mkl.cpp inc/timer.hpp
	CC -mkl $(CFLAGS) -o $@ $< -Iinc

diffusion2d_rw_mkl_run: diffusion2d_rw_mkl_run.cpp inc/timer.hpp
	CC -mkl $(CFLAGS) -o $@ $< -Iinc

diffusion2d_rw_mkl_mt: diffusion2d_rw_mkl_mt.cpp inc/timer.hpp
	CC -mkl $(CFLAGS_THREADS) -o $@ $< -Iinc
	
diffusion2d_rw_mkl_mpi: diffusion2d_rw_mkl_mpi.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -mkl -o $@ $< -Iinc
	
diffusion2d_rw_mpi: diffusion2d_rw_mpi.cpp inc/timer.hpp
	$(CXX) $(CFLAGS) -o $@ $< -Iinc

diffusion2d_rw_mkl_mt_mpi: diffusion2d_rw_mkl_mt_mpi.cpp inc/timer.hpp
	$(CXX) $(CFLAGS_THREADS) -mkl -o $@ $< -Iinc 

clean:
	rm -f diffusion2d_serial  diffusion2d_unroll_8_optim  diffusion2d_unroll_8_mt_run diffusion2d_rw_serial diffusion2d_rw_serial_run diffusion2d_rw_mkl diffusion2d_rw_mkl_run diffusion2d_AVX diffusin2d_AVX_u8 diffusion2d_AVX_align diffusion2d_AVX_mt diffusion2d_AVX_u8_mt diffusion2d_rw_mkl_mt diffusion2d_rw_mkl_mpi diffusion2d_rw_mpi diffusion2d_rw_mkl_mt_mpi 
#diffusion2d_MPI

