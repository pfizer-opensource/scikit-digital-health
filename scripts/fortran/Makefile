ctest : test.c
	gcc -o ctest test.c

ftest : test.f90
	f2py -c -m test test.f90

pfft : pocket_fft.c
	gcc -o pfft pocket_fft.c -lm

testf : pocket_fft.f90
	rm pfi.cp*
	f2py -c -m pfi real_fft.f90 only: execute_real_forward destroy_plan :
	clear
	python t2.py 512

pfi : real_fft.f90
	f2py -c -m pfi real_fft.f90 only: execute_real_forward destroy_plan :
    
cpfi : pocket_fft.c
	python setup.py build_ext -i
	#gcc -Wall -I/home/lukasadamowicz/miniconda3/envs/pfymu/include/python3.8 -I/home/lukasadamowicz/miniconda3/envs/pfymu/lib/python3.8/site-packages/numpy/core/include -shared -fPIC -c pocket_fft.c  -o cpfi.so -L/home/lukasadamowicz/miniconda3/envs/pfymu/lib -lpython3.8 -Wl,-rpath=/home/lukasadamowicz/miniconda3/envs/pfymu/lib -lm
