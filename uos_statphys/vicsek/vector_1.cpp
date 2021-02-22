#include <iostream>
#include <queue>
#include <random>
#include <math.h>
#include <time.h>
#include <fstream>
#include <thread>

#define PI 3.1415926535
#define L 50
#define rho 4
#define N (int)(L*L*rho)
#define S_step (1000)
#define T_step (2000)
#define R 1
#define V 0.03

using namespace std;

random_device rd;
mt19937_64 gen(rd());
uniform_real_distribution<double> my_rand(0.0, 1.0);


void reset_particle(double position_x[], double position_y[], double theta[]) {

	for (int i = 0; i < N; i++) {

		position_x[i] = L * my_rand(gen);
		position_y[i] = L * my_rand(gen);
		theta[i] = 2 * PI * my_rand(gen);

	}

}

void theta_input_queue(double position_x[], double position_y[], double theta[], 
						vector<double> lattice_x[][L / R], vector<double> lattice_y[][L / R],
						vector<double> lattice_theta[][L / R]) {

	for (int i = 0; i < N; i++) {

		lattice_x[(int)(position_x[i] / (R))][(int)(position_y[i] / (R))].push_back(position_x[i]);
		lattice_y[(int)(position_x[i] / (R))][(int)(position_y[i] / (R))].push_back(position_y[i]);
		lattice_theta[(int)(position_x[i] / (R))][(int)(position_y[i] / (R))].push_back(theta[i]);

	}
}

//#topython
void update_theta(double position_x[], double position_y[], double theta[], 
						vector<double> lattice_x[][L / R], vector<double> lattice_y[][L / R],
						vector<double> lattice_theta[][L / R], double eta) {

	int nx = 0;
	int ny = 0;
	double x_prime;
	double y_prime;
	double delta_x;
	double delta_y;
	double sum_sin;
	double sum_cos;
	int total_n;

	int lattice_dx[9] = { -1,0,1,-1,0,1,-1,0,1 };
	int lattice_dy[9] = { -1,-1,-1,0,0,0,1,1,1 };

	double real_distance = 0;

	for (int i = 0; i < N; i++) {

		sum_sin = 0;
		sum_cos = 0;

		for (int j = 0; j < 9; j++) {

			nx = (int)(position_x[i] / (R)+lattice_dx[j] + (L / R)) % (L / R);
			ny = (int)(position_y[i] / (R)+lattice_dy[j] + (L / R)) % (L / R);

			total_n = lattice_x[nx][ny].size();

			for (int loop = 0; loop < total_n; loop++) {

				x_prime = lattice_x[nx][ny][loop];
				y_prime = lattice_y[nx][ny][loop];

				delta_x = position_x[i] - x_prime;
				delta_y = position_y[i] - y_prime;

				if (fabs(delta_x) > L - 2. * R)
				{
					delta_x = L - fabs(delta_x);
				}

				if (fabs(delta_y) > L - 2. * R)
				{
					delta_y = L - fabs(delta_y);
				}

				real_distance = pow(delta_x, 2) + pow(delta_y, 2);

				if (real_distance < pow(R, 2)) {
					sum_sin += sin(lattice_theta[nx][ny][loop]);
					sum_cos += cos(lattice_theta[nx][ny][loop]);
				}

			}
		}

		theta[i] = atan2(sum_sin, sum_cos) + eta * (0.5 - my_rand(gen));

	}

}

void update_position(double position_x[], double position_y[], double theta[]) {

	for (int i = 0; i < N; i++) {

		position_x[i] += V * cos(theta[i]);
		if (position_x[i] > L)  position_x[i] -= L;
		else if (position_x[i] < 0)  position_x[i] += L;

		position_y[i] += V * sin(theta[i]);
		if (position_y[i] > L)  position_y[i] -= L;
		else if (position_y[i] < 0)  position_y[i] += L;

	}

}

void save_OP(double theta[], double order_parameter[], int now_time){
	
	double sum_vx = 0;
	double sum_vy = 0;

	for (int i = 0; i < N; i++) {

		sum_vx += cos(theta[i]);
		sum_vy += sin(theta[i]);

	}

	order_parameter[now_time] = pow(sum_vx * sum_vx + sum_vy * sum_vy, 0.5) / N;

}

void theta_clean_queue(vector<double> lattice_x[][L / R], vector<double> lattice_y[][L / R],
						vector<double> lattice_theta[][L / R]) {

	for (int i = 0; i < L / R; i++) {
		for (int j = 0; j < L / R; j++) {

			lattice_x[i][j].clear();
			lattice_y[i][j].clear();
			lattice_theta[i][j].clear();

		}
	}
}


void single_ensemble(double order_parameter[], double eta, int step) {

	vector<double> lattice_x[L / R][L / R];
	vector<double> lattice_y[L / R][L / R];
	vector<double> lattice_theta[L / R][L / R];

	double position_x[N];
	double position_y[N];
	double theta[N];

	double S_1=0;
	double S_2=0;
	double S_4=0;

	int now_step = step;

	reset_particle(position_x, position_y, theta);

	while (true) {

		now_step++;

		theta_input_queue(position_x, position_y, theta, lattice_x, lattice_y, lattice_theta);
		update_theta(position_x, position_y, theta, lattice_x, lattice_y, lattice_theta, eta);
		update_position(position_x, position_y, theta);
		theta_clean_queue(lattice_x, lattice_y, lattice_theta);

		if (now_step < step + T_step - S_step) continue;
		
		save_OP(theta, order_parameter, now_step - T_step + S_step);

		if (now_step > step + T_step) break;

	}

}

void thread_monteCarlo(double order_parameter[], double eta, int thr_index , int ens_num, int time)
{

	int start_point = thr_index * ens_num * time;

	for (size_t i = 0; i < ens_num; i++)
	{
		single_ensemble(order_parameter, eta, start_point + i * time);
	}

}

int main() {

	clock_t start, end;

	int thr_num = 10;
	int ens_num = 10;

	double order_parameter[10000];
	double eta=3;
	
	start = clock();

	vector<thread> workers;

	int unit = ens_num / thr_num;
	int remain = ens_num % thr_num;

	int save_time = T_step-S_step;

	for (size_t i = 0; i < thr_num; i++) {
		workers.push_back(thread(thread_monteCarlo, order_parameter, eta, i , unit, save_time));
	}
	
	int start_point = unit * thr_num * save_time;

	for (size_t i = 0; i < remain; i++)
	{
		single_ensemble(order_parameter, eta, start_point + i * save_time);	
	}

	for (size_t i = 0; i < thr_num; i++) {
		workers[i].join();
	}

	workers.clear();

	end = clock();

	for (int i =0; i<100;i++){

		cout<< order_parameter[i*100]<< " ";

	}

	cout<<endl<<endl<<end-start<<endl;

}