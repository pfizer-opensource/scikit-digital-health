ctest : test.c
	gcc -o ctest test.c

ftest : test.f90
	f2py -c -m test test.f90

pfft : pocket_fft.c
	gcc -o pfft pocket_fft.c -lm

pfi : pocket_fft.f90
	f2py -c -m pfi pocket_fft.f90 only: execute_real_forward :
