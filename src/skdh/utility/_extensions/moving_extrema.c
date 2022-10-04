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
    stack *eqStack_ext;  // enqueue stack current extrema storage
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
    q->eqStack_ext = newStack(n_items);

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
    free(q->eqStack_ext);
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
    else if (isEmpty(q->eqStack))
    {
        push(q->eqStack, data);
        push(q->eqStack_ext, data);
    }
    else
    {
        push(q->eqStack, data);
        // check what value extrema should take
        double *top;
        peek(q->eqStack_ext, &top);

        // push the current largest value to the extrema stack
        push(q->eqStack_ext, data > *top ? data : *top);
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
    else if (isEmpty(q->eqStack))
    {
        push(q->eqStack, data);
        push(q->eqStack_ext, data);
    }
    else
    {
        push(q->eqStack, data);
        // check what value extrema should take
        double *top;
        peek(q->eqStack_ext, &top);

        // push the current smallest value to the extrema stack
        push(q->eqStack_ext, data < *top ? data : *top);
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
    double max = -9.9E250;  // just make it a very large negative;

    double data;
    double *pr;
    // until enqueue stack is empty
    while (peek(q->eqStack, &pr))
    {
        data = pop(q->eqStack);
        pop(q->eqStack_ext);
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
    double min = 9.9E250;  // make it a very large positive

    double data;
    double *pr;
    // until enqueue stack is empty
    while (peek(q->eqStack, &pr))
    {
        data = pop(q->eqStack);
        pop(q->eqStack_ext);
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
 * Get the current maximum from a queue
 *
 * @param q Queue to get the maximum value from
 */
double get_max(Queue *q)
{
    double *dq_max, *eq_max;
    peek(q->dqStack_ext, &dq_max);
    peek(q->eqStack_ext, &eq_max);
    return *dq_max > *eq_max ? *dq_max : *eq_max;
}

/**
 * Get the current minimum from a queue
 *
 * @param q Queue to get the minimum value from
 */
double get_min(Queue *q)
{
    double *dq_min, *eq_min;
    peek(q->dqStack_ext, &dq_min);
    peek(q->eqStack_ext, &eq_min);
    return *dq_min < *eq_min ? *dq_min : *eq_min;
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
    res[++k] = get_max(q);

    // iterate over the windows
    long ii = *wlen;  // keep track of the last element +1 inserted into the stack
    for (long i = *skip; i < (*n - *wlen + 1); i += *skip)
    {
        for (int j = limax(ii, i); j < i + *wlen; ++j)
        {
            dequeue_max(q);
            enqueue_max(q, x[j]);
        }
        ii = i + *wlen; // update to latest taken element (+1)

        // get the new maximum
        res[++k] = get_max(q);
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
    res[++k] = get_min(q);

    // iterate over the windows
    long ii = *wlen;  // keep track of the last element +1 inserted into the stack
    for (long i = *skip; i < (*n - *wlen + 1); i += *skip)
    {
        for (int j = limax(ii, i); j < i + *wlen; ++j)
        {
            dequeue_min(q);
            enqueue_min(q, x[j]);
        }
        ii = i + *wlen; // update to latest taken element (+1)

        // get the new maximum
        res[++k] = get_min(q);
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
    Queue *q = newQueue(10);

    double x[30];
    srand(50);
    for (int i = 0; i < 30; ++i)
    {
        x[i] = (double)(rand() % 20 + 1);  // between 0 and 19
    }
    // manually generate results
    // wlen = 10
    // skip = 1
    double exp[21];
    for (int j=0; j < 21; ++j)
    {
        exp[j] = x[j];
        for (int i = 0; i < 10; ++i)
        {
            exp[j] = x[j + i] > exp[j] ? x[j + i] : exp[j];
        }
    }

    // prediction
    double pred[21];
    long wlen = 10;
    long skip = 1;
    long n = 30;

    int k = -1;
    for (long i = 0; i < wlen; ++i){
        enqueue_max(q, x[i]);
    }
    pred[++k] = get_extrema(q);

    // iterate windows
    long ii = wlen;
    for (long i = skip; i < (n - wlen + 1); i += skip)
    {
        fprintf(stdout, "[%i] i=%li ii=%li i + wlen=%li\n", k + 1, i, ii, i + wlen);
        for (long j = limax(ii, i); j < i + wlen; ++j)
        {
            fprintf(stdout, "\n");
            dequeue_max(q);
            enqueue_max(q, x[j]);
        }
        ii = i + wlen;  // update the latest taken sample (+ 1)

        // get the new maximum
        pred[++k] = get_extrema(q);
        if (pred[k] != exp[k]){
            fprintf(stdout, "[%i] %f  %f\n", k, pred[k], exp[k]);
            for (int i2 = 0; i2 < wlen; ++i2) {
                fprintf(stdout, "%2.0f, ", x[i + i2]);
            }
            fprintf(stdout, "\n");
            for (int i2 = size(q->dqStack) - 1; i2 >= 0; --i2){
                fprintf(stdout, "%2.0f, ", q->dqStack->items[i2]);
            }
            fprintf(stdout, "|");
            for (int i2 = 0; i2 < size(q->eqStack); ++i2){
                fprintf(stdout, "%2.0f, ", q->eqStack->items[i2]);
            }
            fprintf(stdout, "\n\n");
        }
    }

    // cleanup the queue
    freeQueue(q);

}
*/
