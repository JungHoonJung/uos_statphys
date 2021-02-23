#include "dSFMT.c"
#include "stdio.h"
#include "iostream"
//#include "mkl"
#include "thread"
#include "vector"
#include "random"
#include "cmath"

//including

using namespace std;





//windows용 전처리 구문 dll 파일을 만들때 함수를 잘 내보내기 위한 코드
// 리눅스에서는 쓸모 없음.
/*
#ifndef MONTEI_H
#define MONTEI_H
#define MONTEI_API __declspec(dllexport)
#ifdef __cplusplus
extern "C" {
#endif
    //outcode here
    MONTEI_API void monteCarlo(int* energy, int* mag, double temp, int ens_num, int L, int relax, int time, int thr_num);
    MONTEI_API void wolff(int* energy, int* mag, double temp, int ens_num, int L, int relax, int time, int thr_num);
    MONTEI_API void set_seed(int seed_);

#ifdef __cplusplus
};
#endif
*/

//@extern_python

// random generatior
mt19937 ran_num(1000);
uniform_real_distribution<double> uniform_d(0.0, 1.0);

// random generator (dSFMT)
dsfmt_t dsfmt;
int seed = 12345;

//@python_def
//! set seed of random generator
void set_seed(int seed_) {
    seed = seed_;
    dsfmt_init_gen_rand(&dsfmt, seed);
}


//@python_def
//! function : get sum of nearest neighbor spin state.
int nearest_neighbor(char* conf, int i, int j, int L) {
    return conf[((i + 1) % L) * L + (j % L)] + conf[((i - 1) % L) * L + (j % L)] + conf[((i) % L) * L + ((j + 1) % L)] + conf[((i) % L) * L + ((j - 1) % L)];
}

//@python_def
void MCstep(char* conf, int* index, double* p, int& E, int& M, int L, double temp) {

    for (size_t iter = 0; iter < L * L; iter++) // flip prosedure
    {
        int i = index[iter] / L + L;
        int j = index[iter] % L + L;
        char spin = conf[(i % L) * L + j % L];
        int energy_gap = 2 * spin * nearest_neighbor(conf, i, j, L);
        //if negative energy gap >= 0 spin flip, else compare random number and compare it.
        if (temp != 0. && p[iter] < exp(-energy_gap / temp)) {
            M += -2 * spin;
            conf[(i % L) * L + j % L] *= -1;
            E += energy_gap;
        }
    }
}


//실제 계산될 함수  thread -> work(function)
//계산 결과는 numpy array 하나에 저장하기때문에 single thread에서 범위를 잘 조정하지 않으면
// 같은 곳에 계산 결과 덮어 쓰일수 있음.
// time_min, time_max = 저장할 곳의 인덱스의 시작점과 끝점.
//@python_def
void single_monteCarlo(int* energy, int* mag, double temp, int L, int relax, int time_min, int time_max) {
    //random init 1 or -1
    int timestamp = time_min;
    char* conf = new char[L * L];
    int N = L * L;
    int M;
    if (uniform_d(ran_num) < 0.5) {
        for (size_t i = 0; i < L * L; i++)
        {
            conf[i] = 1;
        }

        M = L * L;

    }
    else {
        for (size_t i = 0; i < L * L; i++)
        {
            conf[i] = -1;
        }
        M = -L * L;
    }
    int E = -2 * L * L;
    uniform_int_distribution<int> uniform_i(0, L * L-1);
    int* indices = new int[L * L];
    double* p = new double[L * L];

    for (size_t i = 0; i < relax; i++) //relaxation
    {
        for (size_t j = 0; j < N; j++)
        {//target and prob. initialization
            indices[j] = uniform_i(ran_num);
            p[j] = uniform_d(ran_num);
        }
        MCstep(conf, indices, p, E, M, L, temp);

    }
    for (size_t i = 0; i < time_max - time_min; i++) //monte-Carlo time average
    {
        for (size_t j = 0; j < N; j++)
        {//target and prob. initialization
            indices[j] = uniform_i(ran_num);
            p[j] = uniform_d(ran_num);
        }
        MCstep(conf, indices, p, E, M, L, temp);
        energy[timestamp] = E;
        mag[timestamp] = M;
        timestamp++;
    }



    delete[] conf;
    delete[] indices;
    delete[] p;
};

//thread 단위로 계산할 함수 실제 한개의 쓰레드가 실행하는 내용
//이 코드에서는 하나의 쓰레드가 시뮬레이션을 여러번 해야하기 때문에 만듬
// 일반적으로 위의 코드가 이 자리에 위치함.
// Parameters
// energy, mag : python에서의 실제 numpy array, 주의할점은 1차원 배열로 만들어서 넘겨줘야함.
// 즉 2d 이면 grid 나눌때와 비슷하게 1차원 배열로 만들고 index//L , index%L 꼴이라고 생각하고 코드 작성.
// etc. 나머지 컨트롤 파라미터 (계산에 쓰일 값).
//@python_def
void thread_monteCarlo(int* energy, int* mag, double temp, int ens_num, int L, int relax, int time, int thr_index) {
    int start_point = thr_index * ens_num * time;

    for (size_t i = 0; i < ens_num; i++)
    {
        single_monteCarlo(energy, mag, temp, L, relax, start_point + i * time, start_point + (i + 1) * time);
    }

}

//master thread 용 함수
//python 에서 직접 호출되는 함수
//main cpu 라고 생각하면 되고 실제로 일을 분배하고
//작업이 다 끝나면 정리까지 해줘야함.
// threading 의 핵심 함수임. 잘 공부할것.
//@python_def
void monteCarlo(int* energy, int* mag, double temp, int ens_num, int L, int relax, int time, int thr_num) {
    //c++ stl container vector : python list 와 유사함. 자료형 고정이라는 차이점이 있음
    //append, pop 기능을 하는 함수가 제공됨.
    vector<thread> workers;
    int unit = ens_num / thr_num;
    int remain = ens_num % thr_num;

    //distribute work
    // 각각의 thread에 어떠한 일을 해줄지 정해주고 array의 어느 부분에 기록해야하는지 지정해서
    // 각 thread에 일을 분배해줌
    for (size_t i = 0; i < thr_num; i++) {
        workers.push_back(thread(thread_monteCarlo, energy, mag, temp, unit, L, relax, time, i));
    }

    // 이 함수도 하나의 쓰레드에서 실행하기 때문에 일의 일부를 실행.
    int start_point = thr_num * unit * time;
    //and remain
    for (size_t i = 0; i < remain; i++)
    {
        single_monteCarlo(energy, mag, temp, L, relax, start_point + i * time, start_point + (i + 1) * time);
    }

    //다른 thread들의 작업이 끝날때까지 대기하고 있다가 모든 thread의 작업이 끝나는 것을 확인함.
    //gather thread
    for (size_t i = 0; i < thr_num; i++) {
        workers[i].join();
    }

    //메모리 해제
    workers.clear();

    //return은 없음. 결과값은 numpy array에 저장됨.
}

//@python_def
void MCwolff(char* conf, int** nn, double threshold, int index, int& E, int& M, int L, double temp) {
    int* cluster = new int[L * L];
    int* cluster_ind = new int[L * L]{ 1, };
    for (size_t i = 0; i < L*L; i++)
    {
        cluster_ind[i] = 1;
    }
    //cluster_ind = {1,};
    int ch = 0;
    //int Msum = 0;
    //int Esum = 0;
    //printf("start index = %d\n", index);
    char spin = conf[index];
    cluster[0] = index;
    cluster_ind[index] = 0;
    //conf[index%(L*L)] *= -1;
    //M += -2 * spin;
    int tem = 0;
    int nn_ind = -1;
    while (ch >= 0) {
        index = cluster[ch--];
        //printf("%d : ", index);
        for (size_t i = 0; i < 4; i++) {
            //printf("\t%d : %d,", nn[index][i], cluster_ind[nn[index][i]]);
            nn_ind = nn[index][i];
            if ( conf[nn_ind]== spin) {
                tem++;
                if (cluster_ind[nn_ind]) {
                    if (dsfmt_genrand_close_open(&dsfmt) < threshold) {
                        cluster[++ch] = nn_ind;
                        cluster_ind[nn_ind] = 0;
                        
                    }
                }
                
            }
            else {
                tem--;
            }
            
        }
        //printf("%d \n", tem);
        conf[index] *= -1;
        M += -2 * spin;

        E += tem * 2;
        tem = 0;

    }
    //printf("step done. E : %d\n", E);
    delete[] cluster;
    delete[] cluster_ind;
}

//실제 계산될 함수  thread -> work(function)
//계산 결과는 numpy array 하나에 저장하기때문에 single thread에서 범위를 잘 조정하지 않으면
// 같은 곳에 계산 결과 덮어 쓰일수 있음.
// time_min, time_max = 저장할 곳의 인덱스의 시작점과 끝점.
//@python_def
void single_wolff(int* energy, int* mag, int** nn_, double temp, int L, int relax, int time_min, int time_max) {
    //random init 1 or -1
    int timestamp = time_min;
    char* conf = new char[L * L];

    int N = L * L;
    int M = 0, E = 0;
    if (temp > 2.269185314213022) {
        for (size_t i = 0; i < L * L; i++) {
            if (uniform_d(ran_num) < 0.5) {


                conf[i] = 1;


                M++;

            }
            else {
                conf[i] = -1;
                M--;
            }
        }
        for (size_t i = 0; i < L * L; i++) {
            for (size_t j = 0; j < 4; j++)
            {
                E += -conf[i] * conf[nn_[i][j]];
            }
            
        }
        E /= 2;
    }
    else {
        E = -2 * L * L;
        if (uniform_d(ran_num) < 0.5) {
            for (size_t i = 0; i < L * L; i++)
            {
                conf[i] = 1;
            }

            M = L * L;

        }
        else {
            for (size_t i = 0; i < L * L; i++)
            {
                conf[i] = -1;
            }
            M = -L * L;
        }
    }
    
    
    double threshold = (temp == 0.) ? 1 : 1 - exp(-2 / temp);
    uniform_int_distribution<int> uniform_i(0, L * L - 1);
    //int* indices = new int[L * L];
    //double* p = new double[L * L];
    //printf("relaxation start\n");
    for (size_t i = 0; i < relax; i++) //relaxation
    {
        
        MCwolff(conf, nn_, threshold, uniform_i(ran_num), E, M, L, temp);
    }
    //printf("simulation start\n");
    for (size_t i = 0; i < time_max - time_min; i++) //monte-Carlo time average for each cluster flip.
    {
        MCwolff(conf, nn_, threshold, uniform_i(ran_num), E, M, L, temp);
        energy[timestamp] = E;
        mag[timestamp] = M;
        timestamp++;
    }



    delete[] conf;
};

//thread 단위로 계산할 함수 실제 한개의 쓰레드가 실행하는 내용
//이 코드에서는 하나의 쓰레드가 시뮬레이션을 여러번 해야하기 때문에 만듬
// 일반적으로 위의 코드가 이 자리에 위치함.
// Parameters
// energy, mag : python에서의 실제 numpy array, 주의할점은 1차원 배열로 만들어서 넘겨줘야함.
// 즉 2d 이면 grid 나눌때와 비슷하게 1차원 배열로 만들고 index//L , index%L 꼴이라고 생각하고 코드 작성.
// etc. 나머지 컨트롤 파라미터 (계산에 쓰일 값).
//@python_def
void thread_wolff(int* energy, int* mag, int** nn_, double temp, int ens_num, int L, int relax, int time, int thr_index) {
    int start_point = thr_index * ens_num * time;
    int** nn = new int* [L * L];
    
    for (size_t i = 0; i < ens_num; i++)
    {
        single_wolff(energy, mag, nn_, temp, L, relax, start_point + i * time, start_point + (i + 1) * time);
    }
    
}


//@python_def
void wolff(int* energy, int* mag, double temp, int ens_num, int L, int relax, int time, int thr_num) {
    //init random number
    dsfmt_init_gen_rand(&dsfmt, seed);
    //share the information of nearest neighbor.
    //printf("L = %d, temp = %f\n", L, temp);
    int** nn = new int* [L * L];
    //printf("nn list:\n");
    for (size_t i = 0; i < L * L; i++) {
        //printf("%d : \n\t", i);
        nn[i] = new int[4];
        nn[i][0] = (i % L) ? (i  - 1) % (L * L) : (i + L - 1) % (L * L);
        nn[i][1] = ((i % L) == L - 1) ? (i + 1 - L )  : (i + 1);
        nn[i][2] = ((i / L) == L - 1) ? i - L * (L - 1) : i + L;
        nn[i][3] = (i / L) ? i - L : i + L * (L - 1);
        /*for (size_t j = 0; j < 4; j++)
        {
            //printf("%d,", nn[i][j]);
            if (nn[i][j] < 0) {
                //printf("less than 0\n");
            }
            if (nn[i][j] > L*L) {
                //printf("greater than system size\n");
            }
        }
        //printf("\n");*/
    }

    

    //c++ stl container vector : python list 와 유사함. 자료형 고정이라는 차이점이 있음
    //append, pop 기능을 하는 함수가 제공됨.
    vector<thread> workers;
    int unit = ens_num / thr_num;
    int remain = ens_num % thr_num;

    //printf("distrubute work.\n");
    //distribute work
    // 각각의 thread에 어떠한 일을 해줄지 정해주고 array의 어느 부분에 기록해야하는지 지정해서
    // 각 thread에 일을 분배해줌
    for (size_t i = 0; i < thr_num; i++) {
        workers.push_back(thread(thread_wolff, energy, mag, nn, temp, unit, L, relax, time, i));
    }

    // 이 함수도 하나의 쓰레드에서 실행하기 때문에 일의 일부를 실행.
    int start_point = thr_num * unit * time;
    //and remain
    for (size_t i = 0; i < remain; i++)
    {
        single_wolff(energy, mag, nn, temp, L, relax, start_point + i * time, start_point + (i + 1) * time);
    }
    //printf("end work.\n");
    //다른 thread들의 작업이 끝날때까지 대기하고 있다가 모든 thread의 작업이 끝나는 것을 확인함.
    //gather thread
    for (size_t i = 0; i < thr_num; i++) {
        workers[i].join();
    }
    //printf("join work.\n");
    //메모리 해제
    workers.clear();

    //return은 없음. 결과값은 numpy array에 저장됨.
    for (size_t i = 0; i < L * L; i++) {
        delete[] nn[i];
    }
    delete[] nn;
    //printf("\tdone.\n");
}
