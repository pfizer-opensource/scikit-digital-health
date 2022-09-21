#include "stack.h"
#include "moving_extrema.h"

/*
 * Queue implementation adopted from:
 * https://stackoverflow.com/questions/4802038/implement-a-queue-in-which-push-rear-pop-front-and-get-min-are-all-consta
 * By Scott Rudiger
 */


// ======================================================================
// utility
// ======================================================================

/**
 * Compute the maximum of two integers
 *
 * @param a, b two values to compare.
 *
 * @result c Bigger of a and b.
 */
long limax(long a, long b)
{
    return a > b? a : b;
}

// ======================================================================
// Data structure for the Queue
// ======================================================================

typedef struct
{
    stack *dqStack;  // dequeue stack
    stack *dqStack_ext;  // dequeue stack extrema storage
    stack *eqStack;  // enqueue stack
} Queue;

/**
 * Initialize a new queue
 *
 * @param n_items Maximum number of items in each of the stacks in the queue
 */
Queue *newQueue(int n_items)
{
    Queue *q = (Queue*)malloc(sizeof(Queue));

    // initialize stacks
    q->dqStack = newStack(n_items);
    q->dqStack_ext = newStack(n_items);
    q->eqStack = newStack(n_items);

    return q;
}

/**
 * Free an initialized queue
 *
 * @param q Queue to free
 */
void freeQueue(Queue *q)
{
    free(q->dqStack);
    free(q->dqStack_ext);
    free(q->eqStack);
    free(q);
}

/**
 * Check if a queue has no elements/values in it
 *
 * @param q Queue to check
 */
int queueIsEmpty(Queue *q)
{
    return isEmpty(q->dqStack) && isEmpty(q->eqStack);
}

/**
 * Add an item/value to the queue in order to calculate a moving maximum
 *
 * @param q Queue to add data to
 * @param data Data to add to the queue
 */
void enqueue_max(Queue *q, double data)
{
    if (queueIsEmpty(q))
    {
        push(q->dqStack, data);
        push(q->dqStack_ext, data);  // current maximum value
    }
    else
    {
        push(q->eqStack, data);
        // check if we need to update the maximum value in the dq stack
        double *next = peek(q->dqStack_ext);
        if (data > *next)  // if larger than, update
        {
            // *next = data;  // for rolling cant just update one item, have to update all
            for (int i = 0; i < size(q->dqStack_ext); i++)
            {
                q->dqStack_ext->items[i] = data;
            }
        }
    }
}

/**
 * Add an item/value to the queue in order to calculate a moving minimum
 *
 * @param q Queue to add data to
 * @param data Data to add to the queue
 */
void enqueue_min(Queue *q, double data)
{
    if (queueIsEmpty(q))
    {
        push(q->dqStack, data);
        push(q->dqStack_ext, data);  // current minimum value
    }
    else
    {
        push(q->eqStack, data);
        // check if we need to update the minimum values in the dq stack
        double *next = peek(q->dqStack_ext);
        if (data < *next) // if smaller than
        {
            // for rolling update all the items so that this minimum is carried for an entire window
            for (int i = 0; i < size(q->dqStack_ext); i++)
            {
                q->dqStack_ext->items[i] = data;
            }
        }
    }
}

/**
 * Move all values from the enqueue stack to the dequeue stack for a moving
 * maximum queue
 *
 * @param q Queue to move values between stacks
 */
void move_all_enqueue_to_dequeue_max(Queue *q)
{
    double max = -9.9E250;  // just make it a very large negative

    // until enqueue stack is empty
    double data;
    while (peek(q->eqStack))
    {
        data = pop(q->eqStack);
        max = data > max ? data : max;  // update max if we need to
        // push data & current maximum
        push(q->dqStack, data);
        push(q->dqStack_ext, max);
    }
}

/**
 * Move all values from the enqueue stack to the dequeue stack for a moving
 * minimum queue
 *
 * @param q Queue to move values between stacks
 */
void move_all_enqueue_to_dequeue_min(Queue *q)
{
    double min = 9.9E250;  // just make very large number instead of infinity
    // until enqueue stack is empty
    double data;
    while (peek(q->eqStack))
    {
        data = pop(q->eqStack);
        min = data < min ? data : min;  // update min if we need to
        // push data & current minimum
        push(q->dqStack, data);
        push(q->dqStack_ext, min);
    }
}

/**
 * Remove/pop a value from the dequeue stack, for a moving maximum queue
 *
 * @param q Queue to pop values from
 */
double dequeue_max(Queue *q)
{
    if (queueIsEmpty(q))
    {
        fprintf(stderr, "Queue is empty, cannot dequeue.\n");
        exit(0);
    }

    double res = pop(q->dqStack);
    pop(q->dqStack_ext);
    // if there is no more data in the dequeue stack, move all data from enqueue stack
    if (isEmpty(q->dqStack))
    {
        move_all_enqueue_to_dequeue_max(q);
    }
    return res;
}

/**
 * Remove/pop a value from the dequeue stack, for a moving minimum queue
 *
 * @param q Queue to pop values from
 */
double dequeue_min(Queue *q)
{
    if (queueIsEmpty(q))
    {
        fprintf(stderr, "Queue is empty, cannot dequeue.\n");
        exit(0);
    }

    double res = pop(q->dqStack);
    pop(q->dqStack_ext);
    // if there is no more data in dequeue stack, move all from enqueue
    if (isEmpty(q->dqStack))
    {
        move_all_enqueue_to_dequeue_min(q);
    }
    return res;
}

/**
 * Get the current maximum/minimum from a queue
 *
 * @param q Queue to get the extrema value from
 */
double get_extrema(Queue *q)
{
    double *next = peek(q->dqStack_ext);
    return *next;
}

// ======================================================================
// Rolling min/max functions
// ======================================================================


/**
 * Compute a rolling/moving maximum across a series of data
 *
 * @param n    Number of elements in `x`
 * @param x    Array of values for which to compute rolling maximum
 * @param wlen Window length, in samples
 * @param skip Window skip, in samples
 * @param res  Array of results
 */
void moving_max_c(long *n, double x[], long *wlen, long *skip, double res[])
{
    // res has (n - wlen) / skip + 1 elements

    // initialize a new queue of wlen length
    Queue *q = newQueue(*wlen);
    int k = -1;  // keeping track of where we are in res

    // push the first wlen elements into the queue
    for (long i = 0; i < *wlen; ++i)
    {
        enqueue_max(q, x[i]);
    }
    // get the maximum of the first window
    res[++k] = get_extrema(q);

    // iterate over the windows
    long ii = *wlen;  // keep track of the last element +1 inserted into the stack
    for (long i = *skip; i < n - wlen + 1; i += *skip)
    {
        for (int j = limax(ii, i); j < i + *wlen - 1; ++j)
        {
            dequeue_max(q);
            enqueue_max(q, x[j]);
        }

        // get the new maximum
        res[++k] = get_extrema(q);
    }

    // cleanup the queue
    freeQueue(q);
}


/**
 * Compute a rolling/moving minimum across a series of data
 *
 * @param n    Number of elements in `x`
 * @param x    Array of values for which to compute rolling minimum
 * @param wlen Window length, in samples
 * @param skip Window skip, in samples
 * @param res  Array of results
 */
void moving_min_c(long *n, double x[], long *wlen, long *skip, double res[])
{
    // res has (n - wlen) / skip + 1 elements

    // initialize a new queue of wlen length
    Queue *q = newQueue(*wlen);
    int k = -1;  // keeping track of where we are in res

    // push the first wlen elements into the queue
    for (long i = 0; i < *wlen; ++i)
    {
        enqueue_min(q, x[i]);
    }
    // get the maximum of the first window
    res[++k] = get_extrema(q);

    // iterate over the windows
    long ii = *wlen;  // keep track of the last element +1 inserted into the stack
    for (long i = *skip; i < n - *wlen + 1; i += *skip)
    {
        for (int j = limax(ii, i); j < i + *wlen - 1; ++j)
        {
            dequeue_min(q);
            enqueue_min(q, x[j]);
        }

        // get the new maximum
        res[++k] = get_extrema(q);
    }

    // cleanup the queue
    freeQueue(q);
}

// ======================================================================
// Testing
// ======================================================================
/*
int main()
{
    Queue *q = newQueue(5);

    double x[17] = {4., 5., 2., 1., 3., 6., 2., 2., 8., 7., 5., 1., 4., 4., 3., 4., 2.};
    // 4, 5, 2, 1, 3 -> 5
    // 5, 2, 1, 3, 6 -> 6
    // 2, 1, 3, 6, 2 -> 6
    // 1, 3, 6, 2, 2 -> 6
    // 3, 6, 2, 2, 8 -> 8
    // 6, 2, 2, 8, 7 -> 8
    // 2, 2, 8, 7, 5 -> 8
    // 2, 8, 7, 5, 1 -> 8
    // 8, 7, 5, 1, 4 -> 8
    // 7, 5, 1, 4, 4 -> 7
    // 5, 1, 4, 4, 3 -> 5
    // 1, 4, 4, 3, 4 -> 4
    // 4, 4, 3, 4, 2 -> 4
    double exp[13] = {5., 6., 6., 6., 8., 8., 8., 8., 8., 7., 5., 4., 4.};

    // first window
    enqueue_max(q, x[0]);
    enqueue_max(q, x[1]);
    enqueue_max(q, x[2]);
    enqueue_max(q, x[3]);
    enqueue_max(q, x[4]);
    fprintf(stdout, "[ 1] max: %1.0f [%1.0f]\n", get_max(q), exp[0]);

    double res;
    for (int i = 5; i < 17; ++i)
    {
        dequeue_max(q, &res);
        enqueue_max(q, x[i]);
        fprintf(stdout, "[%2i] max: %1.0f [%1.0f]   popped: %f\n", i - 3, get_max(q), exp[i - 4], res);
    }
}
*/
