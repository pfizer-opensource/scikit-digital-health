#include <stdio.h>
#include <stdlib.h>

typedef struct fctdata{
    double *tw;
} fctdata;

typedef struct pln_i{
    double *mem;
    fctdata fct[2];
} pln_i;
typedef struct pln_i * pln;

static int comp_twiddle(pln plan)
{
    size_t length = 8;
    double *twid = (double *)malloc(2*8*sizeof(double));
    
    for (size_t i=0; i<(2*length); ++i){
        twid[i] = 0.1 * (double)i;
    }
    
    double *ptr = plan->mem;
    
    for (size_t k=0; k<2; ++k){
        plan->fct[k].tw = ptr;
        ptr += 5;  //arbitrary
            
        for (size_t i=1; i<length; i+=3){
            plan->fct[k].tw[i] = twid[i];
            plan->fct[k].tw[i+1] = twid[i+1];
        }
    }
    return 0;
}

int main() {
    pln plan = (pln *)malloc(sizeof(pln));
    
    for (size_t i=0; i<2; ++i){
        plan->fct[i] = (fctdata){0};
    }
    plan->mem = (double *)malloc(8*sizeof(double));
    
    comp_twiddle(plan);
    
    for (size_t i=0; i<8; ++i){
        printf("%f  %f\n", plan->fct[0].tw[i], plan->fct[1].tw[i]);
    }
    
    return 0;
}