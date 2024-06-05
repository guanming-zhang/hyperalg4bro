#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "hyperalg/base_potential.hpp"
#include "hyperalg/distance.hpp"
#include <vector>

//static const double M_PI = 3.1415926;

namespace ha{

class explicit_CC_potential: public BasePotential{
    static const size_t ndim = 2;
    public:
        const size_t N; // Number of particles
        const double L; // Length of box
        const std::vector<std::vector<double>> K; // Wavevectors (units of 2M_PI)
        const std::vector<double> Sk; // Structure Factor
        const std::vector<double> V; // Potential Weighting
        const size_t pairs;
        std::vector<double> dists;
        const std::vector<double> boxv;
        ha::periodic_distance<ndim> distance;

        explicit_CC_potential(size_t _N, std::vector<std::vector<double>> _K, std::vector<double> _Sk, std::vector<double> _V, double _L)
        : N(_N),
          L(_L),
          K(_K),
          Sk(_Sk),
          V(_V),
          pairs(N*(N-1)/2),
          dists(pairs*2),
          boxv(2, L),
          distance(boxv)
        {}

        inline void update_dists(const std::vector<double>& _points)
        {
            distance.get_pair_distances_vec(_points, dists);
        }

        virtual double get_energy(std::vector<double>& x)
        {
            double C0, Ck, kx, ky, Vk;
            double phi = 0.0;
            update_dists(x);
            // Loop over wavevectors
            for (size_t k = 0; k < K.size(); k++)
            {
                // Wavevector
                kx = K[k][0]*2*M_PI/L;
                ky = K[k][1]*2*M_PI/L;
                // C0 for current wavevector
                C0 = 0.5*N*(Sk[k] - 1.0);
                
                // Calculate C(k)
                Ck = 0.0;
                for (size_t z = 0; z < dists.size(); z += 2)
                {
                    Ck += cos(kx*dists[z] + ky*dists[z+1]);
                }

                // Add to Total Energy
                phi += V[k]*(Ck - C0)*(Ck - C0);
            }
            return phi;
        }

        virtual double get_energy_gradient(std::vector<double>& _points, std::vector<double>& grad)
        {
            double phi = 0.0;
            update_dists(_points);
            for (size_t i = 0; i < grad.size(); i++)
            {
                grad[i] = 0.0;
            }
            double C0;
            double Ck;
            double dC;
            double kx;
            double ky;
            double Vk;
            // Loop over wavevectors
            for (size_t k = 0; k < K.size(); k++)
            {
                // Wavevector
                kx = K[k][0]*2*M_PI/L;
                ky = K[k][1]*2*M_PI/L;
                // C0 for current wavevector
                C0 = 0.5*N*(Sk[k] - 1.0);
                // Calculate C(k)
                Ck = 0.0;
                for (size_t z = 0; z < pairs*2; z += 2)
                {
                    Ck += cos(kx*dists[z] + ky*dists[z+1]);
                }

                // Add to Total Energy
                phi += V[k]*(Ck-C0)*(Ck-C0);

                // Calculate Gradient
                for (size_t x = 0; x < grad.size(); x += 2)
                {
                    dC = 0.0;
                    // dC/dr
                    for (size_t y = 0; y < N*2; y += 2)
                    {
                        dC += sin(kx*(_points[y]-_points[x]) + ky*(_points[y+1]-_points[x+1]));
                    }
                    grad[x]   += V[k]*2*(Ck-C0)*kx*dC;
                    grad[x+1] += V[k]*2*(Ck-C0)*ky*dC;
                }
            }
            return phi;
        }

};

}

// int main()
// {
//     int N = 4;
//     CC_potential* Stealthy = new CC_potential(N, 10, 0, 0, 1);
//     //double init[8] = {0,0,0,0.5,0.5,0,0.5,0.5};
//     double init[8] = {0.1,0.9,0.3,0.4,0.8,0.2,0.6,0.5};
//     //double init[4] = {0,0,0.5,0.5};
//     std::vector<double> initvec (init, init + N*2);
//     std::vector<double> grad (N*2,0);
//     Stealthy->gradient_test(initvec,grad);
//     return 0;
// }
