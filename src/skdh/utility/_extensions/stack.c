#include "stack.h"

/*
 * Stack implementation adopted from:
 * https://www.techiedelight.com/stack-implementation/
 */

// ======================================================================
// Data structure for the stack
// ======================================================================
struct stack
{
    int maxsize;  // max capacity of the stack
    int top;      // Keep track of where in the stack we are
    double *items;  // items in the stack
};

/**
 * Create and initialize a new stack object
 *
 * @param n_items Maximum number of items in the stack
 */
Stack* newStack(int n_items)
{
    Stack *stk = (Stack*)malloc(sizeof(Stack));

    stk->maxsize = n_items;
    stk->top = -1;
    stk->items = (double*)malloc(sizeof(double) * n_items);

    return stk;
}

/**
 * Free an initialized stack
 *
 * @param stk Stack to free associated memory for
 */
void freeStack(Stack *stk)
{
    free(stk->items);
    free(stk);
}

/**
 * utility for returning size of stack
 *
 * @param stk Stack to compute size for
 */
int size(Stack *stk)
{
    return stk->top + 1;
}

/**
 * check if stack is empty or no
 *
 * @param stk Stack to check if empty
 */
int isEmpty(Stack *stk)
{
    return stk->top == -1;
}

/**
 * check if stack is full or not
 *
 * @param stk Stack to check if full
 */
int isFull(Stack *stk)
{
    return stk->top == stk->maxsize - 1;
}

/**
 * Push an element to the stack
 *
 * @param stk Stack to add the element to
 * @param val Value to add to the stack
 */
void push(Stack *stk, double val)
{
    if (!isFull(stk))
    {
        stk->items[++stk->top] = x;
    }
    else
    {
        exit(EXIT_FAILURE);
    }
}

/**
 * Remove an element from the stack
 *
 * @param stk Stack to remove an element from
 */
double pop(Stack *stk)
{
    if (!isEmpty(stk))
    {
        return stk->items[stk->top--];
    }
    else
    {
        exit(EXIT_FAILURE);
    }
}

/**
 * Get the topmost element from a stack
 *
 * @param stk Stack to get value from
 * @param val Storage for the topmost value
 */
int peek(Stack *stk, double **val)
{
    if (!isEmpty(stk))
    {
        *val = &(stk->items[stk->top]);
        return 1;
    }
    else
    {
        return 0;
    }
}