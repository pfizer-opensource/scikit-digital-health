#include "stack.h"

/*
 * Stack implementation adopted from:
 * https://www.techiedelight.com/stack-implementation/
 */

// ======================================================================
// Stack interface methods
// ======================================================================

/**
 * Create and initialize a new stack object
 *
 * @param n_items Maximum number of items in the stack
 */
stack* newStack(int n_items)
{
    stack *stk = (stack*)malloc(sizeof(stack));

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
void freeStack(stack *stk)
{
    free(stk->items);
    free(stk);
}

/**
 * utility for returning size of stack
 *
 * @param stk Stack to compute size for
 */
int size(stack *stk)
{
    return stk->top + 1;
}

/**
 * check if stack is empty or no
 *
 * @param stk Stack to check if empty
 */
int isEmpty(stack *stk)
{
    return stk->top == -1;
}

/**
 * check if stack is full or not
 *
 * @param stk Stack to check if full
 */
int isFull(stack *stk)
{
    return stk->top == stk->maxsize - 1;
}

/**
 * Push an element to the stack
 *
 * @param stk Stack to add the element to
 * @param val Value to add to the stack
 */
void push(stack *stk, double val)
{
    if (!isFull(stk))
    {
        stk->items[++stk->top] = val;
    }
    else
    {
        fprintf(stderr, "Stack is full, cannot push.\n");
        exit(0);
    }
}

/**
 * Remove an element from the stack
 *
 * @param stk Stack to remove an element from
 */
double pop(stack *stk)
{
    return stk->items[stk->top--];
}

/**
 * Get the topmost element from a stack
 *
 * @param stk Stack to get value from
 * @param res Storage for the topmost value
 */
int peek(stack *stk, double **res)
{
    if (!isEmpty(stk))
    {
        *res = &(stk->items[stk->top]);
        return 1;
    }
    else
    {
        return 0;
    }
}
