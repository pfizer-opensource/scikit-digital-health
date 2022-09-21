#ifndef STACK_H_  // guard
#define STACK_H_

#include <stdio.h>
#include <stdlib.h>


typedef struct
{
    int maxsize;  // max capacity of the stack
    int top;      // Keep track of where in the stack we are
    double *items;  // items in the stack
} stack;

// initialization and teardown
stack *newStack(int n_items);
void freeStack(stack *stk);

// utility
int size(stack *stk);
int isEmpty(stack *stk);
int isFull(stack *stk);

// interface with the stack
void push(stack *stk, double val);
double pop(stack *stk);
int peek(stack *stk, double **res);


#endif  // STACK_H_
