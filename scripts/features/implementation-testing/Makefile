pfft : pocket_fft.c
	gcc -o pfft pocket_fft.c -lm

pfi : pocket_fft.f90
	f2py -c -m pfi pocket_fft.f90
    
test : test.f90
	f2py -c -m test test.f90