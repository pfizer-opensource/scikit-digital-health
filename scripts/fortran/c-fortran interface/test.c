#include <stdio.h>

void test_c(int *a){
    printf("inside c-code here...\n");
    
    *a = 135;
}

void test_c2(int *b){
    printf("second function");
    
    *b = -1513;
}