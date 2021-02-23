#include <stdio.h>
#include <iostream>
#include <vector>


int add(int, int);
void array_add(int*, int*, int*, int);
void fibonacci_array(int*, int);

//@extern_python

using namespace std;

// some c++ code

//@python_def
int add(int a, int b){
    return a + b;
}

//@python_def
void array_add(int* c, int* a, int* b, int n){
    for (size_t i = 0; i < n; i++)
    {
        c[i] = a[i] +b[i] + 1;
    }
}

//@python_def
int dot_product(int* a, int* b, int n){
    int sum = 0;
    for (size_t i = 0; i < n; i++)
    {
        sum = sum +  a[i] * b[i];
    }
    return sum;
}



//@python_def
//!Get n-th fibonacci series.
//! Parameters
//! -----------
//! result : nd.array
//!    the array will be filled with fibonacci series by this function.
//! n : int
//!    the length of array
void fibonacci_array(
    int* result, 
    int n
    ){
    for (size_t i = 0; i < n; i++)
    {
        if (i>1){
            result[i] = result[i-1]+result[i-2];
        }
        else{
            result[i] = 1;
        }
    }
}

