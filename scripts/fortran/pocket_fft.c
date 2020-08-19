#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define RALLOC(type,num) \
  ((type *)malloc((num)*sizeof(type)))
#define DEALLOC(ptr) \
  do { free(ptr); (ptr)=NULL; } while(0)

#define SWAP(a,b,type) \
  do { type tmp_=(a); (a)=(b); (b)=tmp_; } while(0)

#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#define WARN_UNUSED_RESULT __attribute__ ((warn_unused_result))
#else
#define NOINLINE
#define WARN_UNUSED_RESULT
#endif

static void copy_and_norm(double *c, double *p1, size_t n, double fct)
  {
  if (p1!=c)
    {
    if (fct!=1.)
      for (size_t i=0; i<n; ++i)
        c[i] = fct*p1[i];
    else
      memcpy (c,p1,n*sizeof(double));
    }
  else
    if (fct!=1.)
      for (size_t i=0; i<n; ++i)
        c[i] *= fct;
  }

// adapted from https://stackoverflow.com/questions/42792939/
// CAUTION: this function only works for arguments in the range [-0.25; 0.25]!
static void my_sincosm1pi (double a, double *restrict res)
  {
  double s = a * a;
  /* Approximate cos(pi*x)-1 for x in [-0.25,0.25] */
  double r =     -1.0369917389758117e-4;
  r = fma (r, s,  1.9294935641298806e-3);
  r = fma (r, s, -2.5806887942825395e-2);
  r = fma (r, s,  2.3533063028328211e-1);
  r = fma (r, s, -1.3352627688538006e+0);
  r = fma (r, s,  4.0587121264167623e+0);
  r = fma (r, s, -4.9348022005446790e+0);
  double c = r*s;
  /* Approximate sin(pi*x) for x in [-0.25,0.25] */
  r =             4.6151442520157035e-4;
  r = fma (r, s, -7.3700183130883555e-3);
  r = fma (r, s,  8.2145868949323936e-2);
  r = fma (r, s, -5.9926452893214921e-1);
  r = fma (r, s,  2.5501640398732688e+0);
  r = fma (r, s, -5.1677127800499516e+0);
  s = s * a;
  r = r * s;
  s = fma (a, 3.1415926535897931e+0, r);
  res[0] = c;
  res[1] = s;
  }

NOINLINE static void calc_first_octant(size_t den, double * restrict res)
  {
  size_t n = (den+4)>>3;
  if (n==0) return;
  res[0]=1.; res[1]=0.;
  if (n==1) return;
  size_t l1=(size_t)sqrt(n);
  for (size_t i=1; i<l1; ++i)
    my_sincosm1pi((2.*i)/den,&res[2*i]);
  size_t start=l1;
  while(start<n)
    {
    double cs[2];
    my_sincosm1pi((2.*start)/den,cs);
    res[2*start] = cs[0]+1.;
    res[2*start+1] = cs[1];
    size_t end = l1;
    if (start+end>n) end = n-start;
    for (size_t i=1; i<end; ++i)
      {
      double csx[2]={res[2*i], res[2*i+1]};
      res[2*(start+i)] = ((cs[0]*csx[0] - cs[1]*csx[1] + cs[0]) + csx[0]) + 1.;
      res[2*(start+i)+1] = (cs[0]*csx[1] + cs[1]*csx[0]) + cs[1] + csx[1];
      }
    start += l1;
    }
  for (size_t i=1; i<l1; ++i)
    res[2*i] += 1.;
  }

NOINLINE static void calc_first_quadrant(size_t n, double * restrict res)
  {
  double * restrict p = res+n;
  calc_first_octant(n<<1, p);
  size_t ndone=(n+2)>>2;
  size_t i=0, idx1=0, idx2=2*ndone-2;
  for (; i+1<ndone; i+=2, idx1+=2, idx2-=2)
    {
    res[idx1]   = p[2*i];
    res[idx1+1] = p[2*i+1];
    res[idx2]   = p[2*i+3];
    res[idx2+1] = p[2*i+2];
    }
  if (i!=ndone)
    {
    res[idx1  ] = p[2*i];
    res[idx1+1] = p[2*i+1];
    }
  }

NOINLINE static void calc_first_half(size_t n, double * restrict res)
  {
  int ndone=(n+1)>>1;
  double * p = res+n-1;
  calc_first_octant(n<<2, p);
  int i4=0, in=n, i=0;
  for (; i4<=in-i4; ++i, i4+=4) // octant 0
    {
    res[2*i] = p[2*i4]; res[2*i+1] = p[2*i4+1];
    }
  for (; i4-in <= 0; ++i, i4+=4) // octant 1
    {
    int xm = in-i4;
    res[2*i] = p[2*xm+1]; res[2*i+1] = p[2*xm];
    }
  for (; i4<=3*in-i4; ++i, i4+=4) // octant 2
    {
    int xm = i4-in;
    res[2*i] = -p[2*xm+1]; res[2*i+1] = p[2*xm];
    }
  for (; i<ndone; ++i, i4+=4) // octant 3
    {
    int xm = 2*in-i4;
    res[2*i] = -p[2*xm]; res[2*i+1] = p[2*xm+1];
    }
  }

NOINLINE static void fill_first_quadrant(size_t n, double * restrict res)
  {
  const double hsqt2 = 0.707106781186547524400844362104849;
  size_t quart = n>>2;
  if ((n&7)==0)
    res[quart] = res[quart+1] = hsqt2;
  for (size_t i=2, j=2*quart-2; i<quart; i+=2, j-=2)
    {
    res[j  ] = res[i+1];
    res[j+1] = res[i  ];
    }
  }

NOINLINE static void fill_first_half(size_t n, double * restrict res)
  {
  size_t half = n>>1;
  if ((n&3)==0)
    for (size_t i=0; i<half; i+=2)
      {
      res[i+half]   = -res[i+1];
      res[i+half+1] =  res[i  ];
      }
  else
    for (size_t i=2, j=2*half-2; i<half; i+=2, j-=2)
      {
      res[j  ] = -res[i  ];
      res[j+1] =  res[i+1];
      }
  }

NOINLINE static void fill_second_half(size_t n, double * restrict res)
  {
  if ((n&1)==0)
    for (size_t i=0; i<n; ++i)
      res[i+n] = -res[i];
  else
    for (size_t i=2, j=2*n-2; i<n; i+=2, j-=2)
      {
      res[j  ] =  res[i  ];
      res[j+1] = -res[i+1];
      }
  }


NOINLINE static void sincos_2pibyn_half(size_t n, double * restrict res)
  {
  if ((n&3)==0)
    {
    calc_first_octant(n, res);
    fill_first_quadrant(n, res);
    fill_first_half(n, res);
    }
  else if ((n&1)==0)
    {
    calc_first_quadrant(n, res);
    fill_first_half(n, res);
    }
  else
    calc_first_half(n, res);
  }

NOINLINE static void sincos_2pibyn(size_t n, double * restrict res)
  {
  sincos_2pibyn_half(n, res);
  fill_second_half(n, res);
  }

NOINLINE static size_t largest_prime_factor (size_t n)
  {
  size_t res=1;
  size_t tmp;
  while (((tmp=(n>>1))<<1)==n)
    { res=2; n=tmp; }

  size_t limit=(size_t)sqrt(n+0.01);
  for (size_t x=3; x<=limit; x+=2)
  while (((tmp=(n/x))*x)==n)
    {
    res=x;
    n=tmp;
    limit=(size_t)sqrt(n+0.01);
    }
  if (n>1) res=n;

  return res;
  }

NOINLINE static double cost_guess (size_t n)
  {
  const double lfp=1.1; // penalty for non-hardcoded larger factors
  size_t ni=n;
  double result=0.;
  size_t tmp;
  while (((tmp=(n>>1))<<1)==n)
    { result+=2; n=tmp; }

  size_t limit=(size_t)sqrt(n+0.01);
  for (size_t x=3; x<=limit; x+=2)
  while ((tmp=(n/x))*x==n)
    {
    result+= (x<=5) ? x : lfp*x; // penalize larger prime factors
    n=tmp;
    limit=(size_t)sqrt(n+0.01);
    }
  if (n>1) result+=(n<=5) ? n : lfp*n;

  return result*ni;
  }

/* returns the smallest composite of 2, 3, 5, 7 and 11 which is >= n */
NOINLINE static size_t good_size(size_t n)
  {
  if (n<=6) return n;

  size_t bestfac=2*n;
  for (size_t f2=1; f2<bestfac; f2*=2)
    for (size_t f23=f2; f23<bestfac; f23*=3)
      for (size_t f235=f23; f235<bestfac; f235*=5)
        for (size_t f2357=f235; f2357<bestfac; f2357*=7)
          for (size_t f235711=f2357; f235711<bestfac; f235711*=11)
            if (f235711>=n) bestfac=f235711;
  return bestfac;
  }



#define NFCT 25

struct rfft_plan_i;
typedef struct rfft_plan_i * rfft_plan;

typedef struct cmplx {
  double r,i;
} cmplx;

typedef struct rfftp_fctdata
  {
  size_t fct, twsize;
  double *tw, *tws;
  } rfftp_fctdata;

typedef struct rfftp_plan_i
  {
  size_t length, nfct, twsize;
  double *mem;
  rfftp_fctdata fct[NFCT];
  } rfftp_plan_i;
typedef struct rfftp_plan_i * rfftp_plan;


typedef struct rfft_plan_i
  {
  rfftp_plan packplan;
  } rfft_plan_i;

/* -----------------------------------------------
--------------
-------------
-----------
--------
---- */


WARN_UNUSED_RESULT
static int rfftp_factorize (rfftp_plan plan)
  {
  size_t length=plan->length;
  size_t nfct=0;
  while ((length%4)==0)
    { if (nfct>=NFCT) return -1; plan->fct[nfct++].fct=4; length>>=2; }
  if ((length%2)==0)
    {
    length>>=1;
    // factor 2 should be at the front of the factor list
    if (nfct>=NFCT) return -1;
    plan->fct[nfct++].fct=2;
    SWAP(plan->fct[0].fct, plan->fct[nfct-1].fct,size_t);
    }
  size_t maxl=(size_t)(sqrt((double)length))+1;
  for (size_t divisor=3; (length>1)&&(divisor<maxl); divisor+=2)
    if ((length%divisor)==0)
      {
      while ((length%divisor)==0)
        {
        if (nfct>=NFCT) return -1;
        plan->fct[nfct++].fct=divisor;
        length/=divisor;
        }
      maxl=(size_t)(sqrt((double)length))+1;
      }
  if (length>1) plan->fct[nfct++].fct=length;
  plan->nfct=nfct;
  return 0;
  }

static size_t rfftp_twsize(rfftp_plan plan)
  {
  size_t twsize=0, l1=1;
  for (size_t k=0; k<plan->nfct; ++k)
    {
    size_t ip=plan->fct[k].fct, ido= plan->length/(l1*ip);
    twsize+=(ip-1)*(ido-1);
    if (ip>5) twsize+=2*ip;
    l1*=ip;
    }
  return twsize;
  return 0;
  }

WARN_UNUSED_RESULT NOINLINE static int rfftp_comp_twiddle (rfftp_plan plan)
  {
  size_t length=plan->length;
  double *twid = RALLOC(double, 2*length);
  if (!twid) return -1;
  sincos_2pibyn_half(length, twid);
  size_t l1=1, iptr;
  double *ptr=plan->mem;

  iptr = 0;
    
  for (size_t k=0; k<plan->nfct; ++k)
    {
    size_t ip=plan->fct[k].fct, ido=length/(l1*ip);
    if (k<plan->nfct-1) // last factor doesn't need twiddles
      {
      plan->fct[k].tw = ptr;
      plan->fct[k].twsize = plan->twsize - iptr;
      ptr += (ip-1)*(ido-1);
      iptr += (ip-1)*(ido-1);
      for (size_t j=1; j<ip; ++j){
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          plan->fct[k].tw[(j-1)*(ido-1)+2*i-2] = twid[2*j*l1*i];
          plan->fct[k].tw[(j-1)*(ido-1)+2*i-1] = twid[2*j*l1*i+1];
          }
       }
      }
    if (ip>5) // special factors required by *g functions
      {
      plan->fct[k].tws=ptr; ptr+=2*ip;
      plan->fct[k].tws[0] = 1.;
      plan->fct[k].tws[1] = 0.;
      for (size_t i=1; i<=(ip>>1); ++i)
        {
        plan->fct[k].tws[2*i  ] = twid[2*i*(length/ip)];
        plan->fct[k].tws[2*i+1] = twid[2*i*(length/ip)+1];
        plan->fct[k].tws[2*(ip-i)  ] = twid[2*i*(length/ip)];
        plan->fct[k].tws[2*(ip-i)+1] = -twid[2*i*(length/ip)+1];
        }
      }
    l1*=ip;
    }
  DEALLOC(twid);
  return 0;
  }

NOINLINE static rfftp_plan make_rfftp_plan (size_t length)
  {
  if (length==0) return NULL;
  rfftp_plan plan = RALLOC(rfftp_plan_i,1);
  if (!plan) return NULL;
  plan->length=length;
  plan->nfct=0;
  plan->mem=NULL;
  for (size_t i=0; i<NFCT; ++i)
    plan->fct[i]=(rfftp_fctdata){0,0,0};
  if (length==1) return plan;
  if (rfftp_factorize(plan)!=0) { DEALLOC(plan); return NULL; }
  size_t tws=rfftp_twsize(plan);
  
  plan->twsize = tws;
    
  plan->mem=RALLOC(double,tws);
  if (!plan->mem) { DEALLOC(plan); return NULL; }
  if (rfftp_comp_twiddle(plan)!=0)
    { DEALLOC(plan->mem); DEALLOC(plan); return NULL; }
  return plan;
  }

NOINLINE static void destroy_rfftp_plan (rfftp_plan plan)
  {
  DEALLOC(plan->mem);
  DEALLOC(plan);
  }



#define WA(x,i) wa[(i)+(x)*(ido-1)]
#define PM(a,b,c,d) { a=c+d; b=c-d; }
/* (a+ib) = conj(c+id) * (e+if) */
#define MULPM(a,b,c,d,e,f) { a=c*e+d*f; b=c*f-d*e; }

#define CC(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+cdim*(c))]

NOINLINE static void radf2 (size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=2;

  for (size_t k=0; k<l1; k++)
    PM (CH(0,0,k),CH(ido-1,1,k),CC(0,k,0),CC(0,k,1))
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(    0,1,k) = -CC(ido-1,k,1);
      CH(ido-1,0,k) =  CC(ido-1,k,0);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double tr2, ti2;
      MULPM (tr2,ti2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      PM (CH(i-1,0,k),CH(ic-1,1,k),CC(i-1,k,0),tr2)
      PM (CH(i  ,0,k),CH(ic  ,1,k),ti2,CC(i  ,k,0))
      }
  }

NOINLINE static void radf4(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=4;
  static const double hsqt2=0.70710678118654752440;

  for (size_t k=0; k<l1; k++)
    {
    double tr1,tr2;
    PM (tr1,CH(0,2,k),CC(0,k,3),CC(0,k,1))
    PM (tr2,CH(ido-1,1,k),CC(0,k,0),CC(0,k,2))
    PM (CH(0,0,k),CH(ido-1,3,k),tr2,tr1)
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      double ti1=-hsqt2*(CC(ido-1,k,1)+CC(ido-1,k,3));
      double tr1= hsqt2*(CC(ido-1,k,1)-CC(ido-1,k,3));
      PM (CH(ido-1,0,k),CH(ido-1,2,k),CC(ido-1,k,0),tr1)
      PM (CH(    0,3,k),CH(    0,1,k),ti1,CC(ido-1,k,2))
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      MULPM(cr2,ci2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      MULPM(cr3,ci3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2))
      MULPM(cr4,ci4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3))
      PM(tr1,tr4,cr4,cr2)
      PM(ti1,ti4,ci2,ci4)
      PM(tr2,tr3,CC(i-1,k,0),cr3)
      PM(ti2,ti3,CC(i  ,k,0),ci3)
      PM(CH(i-1,0,k),CH(ic-1,3,k),tr2,tr1)
      PM(CH(i  ,0,k),CH(ic  ,3,k),ti1,ti2)
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr3,ti4)
      PM(CH(i  ,2,k),CH(ic  ,1,k),tr4,ti3)
      }
  }

#undef CC
#undef CH
#undef C1
#undef C2
#undef CH2

WARN_UNUSED_RESULT
static int rfftp_forward(rfftp_plan plan, double c[], double fct){
    if (plan->length==1) return 0;
    size_t n=plan->length;
    size_t l1=n, nf=plan->nfct;
    double *ch = RALLOC(double, n);
    if (!ch) return -1;
    double *p1=c, *p2=ch;

    for(size_t k1=0; k1<nf;++k1){
        size_t k=nf-k1-1;
        size_t ip=plan->fct[k].fct;
        size_t ido=n / l1;
        l1 /= ip;
        
        if(ip==4)
            radf4(ido, l1, p1, p2, plan->fct[k].tw);
        else if(ip==2)
            radf2(ido, l1, p1, p2, plan->fct[k].tw);
        else{
            return -1;
        }
        SWAP (p1,p2,double *);
        }
        copy_and_norm(c,p1,n,fct);
        DEALLOC(ch);
        return 0;
  }

/* -----------------------------------------------
--------------
-------------
-----------
--------
---- */
int main (int argc, char *argv[]) {
    rfftp_plan plan = NULL;
    if (argc == 2){
        plan = make_rfftp_plan(atoi(argv[1]));
    } else {
        plan = make_rfftp_plan(64);
    }
    
    //rfftp_forward(plan);
    
    printf("length: %lu\n", plan->length);
    printf("nfct: %lu\n", plan->nfct);
    
    for (int i=0; i<plan->nfct; ++i){
        printf("fct %d: %lu  \n", i, plan->fct[i].fct);

/*
        if (plan->fct[i].tw){
            for (size_t j=0; j<plan->fct[i].twsize; ++j){
                printf("%6.2f", plan->fct[i].tw[j]);
                if ((j+1)%10==0) printf("\n");
            }
            printf("\n");
        }
*/
    }
    
    size_t tws=rfftp_twsize(plan);
    
    return 0;
}
