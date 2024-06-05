#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "hyperalg/base_potential.hpp"
#include "hyperalg/distance.hpp"
#include <vector>

//static const double M_PI = 3.1415926;

namespace ha{

class CC_potential: public BasePotential{
    static const size_t ndim = 2;
    public:
        const size_t N; // Number of particles
        const double L; // Length of box
        const double K; // Wavevector threshhold (units of 2M_PI)
        const double a; // Power Law exponent
        const double D;
        const size_t pairs;
        std::vector<double> dists;
        const std::vector<double> boxv;
        ha::periodic_distance<ndim> distance;

        CC_potential(size_t _N, double _K, double _a, double _DKa, double _L)
        : N(_N),
          L(_L),
          K(_K),
          a(_a),
          D(_DKa/pow(K*2*M_PI,a)),
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
            double C0, Ck, kx, ky;
            double phi = 0.0;
            update_dists(x);
            // Loop over wavevectors
            int Nk = 0;
            for (int i = -K; i <= K; i++)
            {
                for (int j = -K; j <= K; j++)
                {
                    // Check that wavevector magnitude is < K
                    // Check that it is not the zero wavevector
                    if (i*i + j*j < K*K && (i != 0 || j != 0))
                    {
                        Nk++;
                        // Wavevector
                        kx = i*2*M_PI/L;
                        ky = j*2*M_PI/L;

                        // C0 for current wavevector
                        C0 = -0.5*N;
                        if (a > 0)
                        {
                            C0 += D*pow(kx*kx + ky*ky, 0.5*a);
                        }

                        // Calculate C(k)
                        Ck = 0.0;
                        for (size_t z = 0; z < dists.size(); z += 2)
                        {
                            Ck += cos(kx*dists[z] + ky*dists[z+1]);
                        }

                        // Add to Total Energy
                        phi += (Ck - C0)*(Ck - C0);
                    }
                }
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
            // Loop over wavevectors
            int Nk = 0;
            for (int i = -K; i <= K; i++)
            {
                for (int j = -K; j <= K; j++)
                {
                    // Check that wavevector magnitude is < K
                    // Check that it is not the zero wavevector
                    if (i*i + j*j < K*K && (i != 0 || j != 0))
                    {
                        Nk++;
                        // Wavevector
                        kx = i*2.*M_PI/L;
                        ky = j*2.*M_PI/L;
                        // C0 for current wavevector
                        C0 = -0.5*N;
                        if (a > 0)
                        {
                            C0 += D*pow(kx*kx + ky*ky, 0.5*a);
                        }
                        // Calculate C(k)
                        Ck = 0.0;
                        for (size_t z = 0; z < pairs*2; z += 2)
                        {
                            Ck += cos(kx*dists[z] + ky*dists[z+1]);
                        }
                        std::cout << Ck;

                        // Add to Total Energy
                        phi += (Ck-C0)*(Ck-C0);

                        // Calculate Gradient
                        for (size_t x = 0; x < grad.size(); x += 2)
                        {
                            dC = 0.0;
                            // dC/dr
                            for (size_t y = 0; y < N*2; y += 2)
                            {
                                dC += sin(kx*(_points[y]-_points[x]) + ky*(_points[y+1]-_points[x+1]));
                            }
                            //std::cout << dC << "\n";
                            grad[x]   += 2*(Ck-C0)*kx*dC;
                            grad[x+1] += 2*(Ck-C0)*ky*dC;
                        }
                    }
                }
            }
            std::cout << Nk << '\n';
            return phi;
        }

        void gradient_test(std::vector<double> _points, std::vector<double>& grad)
        {
            std::cout << "Energy: " << get_energy_gradient(_points,grad)<<"\nGrad:     ";
            for (size_t i=0; i<N*2; i++)
            {
                std::cout << grad[i] << ", ";
            }
            std::cout << '\n';
            numerical_gradient(_points, grad);
            std::cout << "Num Grad: ";
            for (size_t i=0;i<N*2;i++)
            {
                std::cout << grad[i] << ", ";
            }
            std::cout << '\n';
        
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
