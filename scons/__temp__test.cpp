#include <stdio.h>
#include <iostream>

extern "C"{
__declspec(dllexport) int add(int a, int b);
__declspec(dllexport) void array_add(int* c, int* a, int* b, int n);
};

using namespace std;



//@python_def
int add(int a, int b){
    return a + b;
}

//@python_def
void array_add(int* c, int* a, int* b, int n){
    for (size_t i = 0; i < n; i++)
    {
        c[i] = a[i] +b[i];
    }
}
