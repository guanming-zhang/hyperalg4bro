#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "hyperalg/base_potential.hpp"
#include "hyperalg/distance.hpp"
#include "hyperalg/meta_pow.hpp"
#include <vector>

namespace ha{

class Hertzian_potential: public BasePotential{
    static const size_t ndim = 3;
    public:
	const size_t N; // Number of particles
        const double L; // Length of box
        const double a; // Inverse power exponent
        const double eps;
        const size_t pairs;
        std::vector<double> dists;
        const std::vector<double> boxv;
	const std::vector<double> radii;
        const std::vector<int> pairidx;
        ha::periodic_distance<ndim> distance;

        Hertzian_potential(size_t _N, double _pow, double _eps, std::vector<double> _radii, double _L)
        : N(_N),
          L(_L),
          a(_pow),
          eps(_eps),
	  radii(_radii),
          pairs(N*(N-1)/2),
          dists(pairs),
          boxv(ndim, L),
          distance(boxv),
          pairidx(get_pair_indices())
        {}

        std::vector<int> get_pair_indices() 
        {
            std::vector<int> idx;
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    idx.push_back(i);
                    idx.push_back(j);
                }
            }
            //pairidx = idx;
            return idx;
        }

        inline void update_dists(const std::vector<double>& x)
        {
            distance.get_pair_distances(x, dists);
        }

        virtual double get_energy(std::vector<double>& x)
        {
            double phi = 0.0;
	    double radius_sum;
	    int idxi, idxj;
            update_dists(x);
            // Loop over distances
            for (size_t i = 0; i < pairs; i++)
            {
                // indices for atoms i & j
                idxi = pairidx[i*2];
                idxj = pairidx[i*2+1];
                radius_sum = radii[idxi] + radii[idxj];
                if (dists[i] < radius_sum)
                {
                    phi += std::pow((1 -dists[i]/radius_sum), a) * eps/a;
                }
            }
            return phi;
        }

        virtual double get_energy_gradient(std::vector<double>& x, std::vector<double>& grad)
        {
            double phi = 0.0;
            double factor, radius_sum;
            double dr[ndim];
            int idxi,idxj;
            update_dists(x);
            grad.assign(grad.size(),0.);
            // Loop over distances
	    
	    
	    for (size_t i = 0; i < pairs; i++)
            {
                // indices for atoms i & j
                idxi = pairidx[i*2];
                idxj = pairidx[i*2+1];
                radius_sum = radii[idxi] + radii[idxj];
		if (dists[i] < radius_sum)
                {
                    factor = std::pow((1 -dists[i]/radius_sum), a) * eps;
                    phi += factor/a;
                    factor /= (dists[i]-radius_sum)*dists[i];
		    idxi *= ndim;
		    idxj *= ndim;
                    distance.get_rij(dr, &x[idxj], &x[idxi]);
                    // Contributions to gradients of i & j
                    for (size_t k=0; k<ndim; ++k) {
                        grad[idxi+k] -= factor * dr[k];
                        grad[idxj+k] += factor * dr[k];
                    }
                }
            }
            return phi;
        }

        void gradient_test(std::vector<double> x, std::vector<double>& grad)
        {
            std::cout << "Energy: " << get_energy_gradient(x,grad)<<"\nGrad:     ";
            for (size_t i=0; i<N*2; i++)
            {
                std::cout << grad[i] << ", ";
            }
            std::cout << '\n';
            numerical_gradient(x, grad);
            std::cout << "Num Grad: ";
            for (size_t i=0;i<N*2;i++)
            {
                std::cout << grad[i] << ", ";
            }
            std::cout << "\n\nDistances: ";
            for (size_t i=0;i<pairs;i++)
            {
                std::cout << dists[i] << ", ";
            }
        }
        

};

}
