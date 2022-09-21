#ifndef STACK_H_  // guard
#define STACK_H_


typedef struct stack Stack;

// initialization and teardown
Stack *newStack(int n_items);
void freeStack(Stack *stk);

// utility
int size(Stack *stk);
int isEmpty(Stack *stk);
int isFull(Stack *stk);

// interface with the stack
void push(Stack *stk, double val);
double pop(Stack *stk);
int peek(Stack *stk, double **val);


#endif  // STACK_H_
