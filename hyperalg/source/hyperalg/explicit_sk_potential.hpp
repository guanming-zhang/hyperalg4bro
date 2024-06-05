#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "finufft.h"
#include "base_potential.hpp"
#include <vector>
#include <complex>

//static const double M_PI = 3.1415926;
static const std::complex<double> I(0.0,1.0);

namespace ha{

class explicit_Sk_potential: public BasePotential{
    //static const size_t ndim = 2;
    public:
        const size_t ndim; //dimension
        const size_t N; // Number of particles
        const std::vector<double> L; // box dimensions
        const double K; // max K magnitude to constrain
        const double beta; // Error threshhold sharpness
        const double gamma; // Error threshhold dilation
        const std::vector<int> Kvec; // Wavevectors (units of 2M_PI)
        const std::vector<double> Kmag; // Wavevector magnitudes (units of 2M_PI)
        const std::vector<double> Sk0; // Structure Factor
        const std::vector<double> V; // Potential Weighting
        std::vector<std::complex<double>> c; // Point weights
        const int error_mode; //Form of U(k) to be used

        explicit_Sk_potential(std::vector<double> radii, double _K, std::vector<double> _Sk, std::vector<double> _V, std::vector<double> _L, double _beta, double _gamma, int _error_mode)
        : N(radii.size()),
          L(_L),
          ndim(_L.size()),
          K(_K),
          Kvec(calculate_Kvec(K)),
          Kmag(calculate_Kmag(Kvec)),
          Sk0(_Sk),
          V(_V),
          c(initialize_c(radii)),
          beta(_beta),
          gamma(_gamma),
          error_mode(_error_mode)
        {}

        std::vector<int> calculate_Kvec(double K)
        {
            std::vector<int> _Kvec(2*size_t(K)+1);
            for (size_t i=0; i < _Kvec.size(); i++)
	    {
		_Kvec[i] = int(i)-int(K);
	    }
            return _Kvec;
        }

        std::vector<double> calculate_Kmag(std::vector<int> Kvec)
        {
            if(ndim == 2)
            {
                std::vector<double> _Kmag(Kvec.size()*Kvec.size());
                for (size_t i=0; i < Kvec.size(); i++)
   	        {
                    for(size_t j=0; j < Kvec.size(); j++)
		    {
                        _Kmag[i+Kvec.size()*j] = sqrt(Kvec[i]*Kvec[i]+Kvec[j]*Kvec[j]);
                    }
	        }
                return _Kmag;
            }
            else if(ndim == 3)
            {
                std::vector<double> _Kmag(Kvec.size()*Kvec.size()*Kvec.size());
                for (size_t i=0; i < Kvec.size(); i++)
   	        {
                    for(size_t j=0; j < Kvec.size(); j++)
		    {     
                        for(size_t k=0; k < Kvec.size(); k++)
                        {
                            _Kmag[i+Kvec.size()*j+Kvec.size()*Kvec.size()*k] = sqrt(Kvec[i]*Kvec[i]+Kvec[j]*Kvec[j]+Kvec[k]*Kvec[k]);
                        }
                    }
	        }
                return _Kmag;
            }
            else
            {
                 throw std::runtime_error("Invalid dimension: "+std::to_string(ndim));
            }
        }

        std::vector<std::complex<double>> initialize_c(std::vector<double> radii)
        {
            std::vector<std::complex<double>> _c(radii.size(),1);
            double sum = 0;
            for (size_t i=0; i < _c.size(); i++)
	    {
                for (size_t j=0; j<ndim; j++)
		{
                    if(ndim ==2)
                    {
                        _c[i] = _c[i]*M_PI*radii[i]*radii[i];
                    }
                    else if(ndim == 3)
                    {
                        _c[i] = _c[i]*M_PI*radii[i]*radii[i]*radii[i]*4.0/3.0;
                    }
                }
                sum += std::real(_c[i]);
	    } 
            for (size_t i=0; i < _c.size(); i++)
	    {
                _c[i] *= _c.size()/sum;
	    }
            return _c;
        }

        void update_c(std::vector<double> radii)
        {
	    c.assign(c.size(),1);
            double sum = 0;
            for (size_t i=0; i < c.size(); i++)
	    {
                for (size_t j=0; j<ndim; j++)
		{
                    c[i] = c[i]*radii[i];
                }
                sum += std::real(c[i]);
	    } 
            for (size_t i=0; i < c.size(); i++)
	    {
                c[i] *= c.size()/sum;
	    }
            return;
        }

        void calculate_U(double& U, double& dU, double Skdiff2, double Kval)
        {
            if(error_mode == 0)
            {
                U = 1;
                dU = 0;
                return;
            }
            U = 1/(1+exp(beta*(gamma*(1+Kval/K)-Skdiff2)));
            dU = beta*U*(1-U);
            return;
        }

        virtual double get_energy(std::vector<double>& points)
        {
            if(ndim == 2)
            {
                return get_energy_2d(points);
            }
            else if(ndim ==3)
            {
                return get_energy_3d(points);
            }
            else
            {
                throw std::runtime_error("Invalid dimension");
            }
        }

        virtual double get_energy_gradient(std::vector<double>& points, std::vector<double>& grad)
        {
            if(ndim == 2)
            {
                return get_energy_gradient_2d(points, grad);
            }
            else if(ndim ==3)
            {
                return get_energy_gradient_3d(points,grad);
            }
            else
            {
                throw std::runtime_error("Invalid dimension");
            }
        }
        double get_energy_2d(std::vector<double>& points)
        {
            std::vector<double> x(N); // x coordinate
            std::vector<double> y(N); // y coordinate
	    std::vector<std::complex<double>> rho(Kmag.size());
            double Skdiff, Skdiff2, U, dU;
            double phi = 0.0;
	    int Nk = Kvec.size();
	    for (size_t j = 0; j < N; j++)
            {
		x[j] = (points[2*j]-round(points[2*j]/L[0])+L[0])*2*M_PI/L[0];
		y[j] = (points[2*j+1]-round(points[2*j+1]/L[1])+L[1])*2*M_PI/L[1];
	    }
	    finufft2d1(N, &x[0], &y[0], &c[0], +1, 1e-6, Nk, Nk, &rho[0], NULL);
	    for (size_t i = 0; i < rho.size(); i++)
	    {
		Skdiff = std::real(std::abs(rho[i]));
		Skdiff = Skdiff*Skdiff/N - Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
		phi += V[i]*U*Skdiff2;
	    }

            return phi;
        }
        double get_energy_3d(std::vector<double>& points)
        {
            std::vector<double> x(N); // x coordinate
            std::vector<double> y(N); // y coordinate
            std::vector<double> z(N); // z coordinate
	    std::vector<std::complex<double>> rho(Kmag.size());
            double Skdiff, Skdiff2, U, dU;
            double phi = 0.0;
	    
	    int Nk = Kvec.size();
	    for (size_t j = 0; j < N; j++)
            {
		x[j] = (points[3*j]-round(points[3*j]/L[0])+L[0])*2*M_PI/L[0];
		y[j] = (points[3*j+1]-round(points[3*j+1]/L[1])+L[1])*2*M_PI/L[1];
		z[j] = (points[3*j+2]-round(points[3*j+2]/L[2])+L[2])*2*M_PI/L[2];
	    }
            
	    finufft3d1(N, &x[0], &y[0], &z[0], &c[0], +1, 1e-6, Nk, Nk, Nk, &rho[0], NULL);
            
	    for (size_t i = 0; i < rho.size(); i++)
	    {
		Skdiff = std::real(std::abs(rho[i]));
		Skdiff = Skdiff*Skdiff/N - Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
		phi += V[i]*U*Skdiff2;
	    }
            return phi;
        }

        double get_energy_gradient_2d(std::vector<double>& points, std::vector<double>& grad)
        {
            std::vector<double> x(N); // x coordinate
            std::vector<double> y(N); // y coordinate
	    std::vector<std::complex<double>> rho(Kmag.size());
            double Skdiff, Skdiff2, U, dU;
	    std::complex<double> factor;
            double phi = 0.0;
	    std::complex<double> Ifactor(0.0,-4.0/N);
	    grad.assign(grad.size(),0);
	    
	    int Nk = Kvec.size();
	    for (size_t j = 0; j < N; j++)
            {
		x[j] = (points[2*j]-round(points[2*j]/L[0])+L[0])*2*M_PI/L[0];
		y[j] = (points[2*j+1]-round(points[2*j+1]/L[1])+L[1])*2*M_PI/L[1];
	    }
            
	    finufft2d1(N, &x[0], &y[0], &c[0], +1, 1e-6, Nk, Nk, &rho[0], NULL);
            
	    std::vector<std::complex<double>> fx(rho.size()), fy(rho.size()), cx(N), cy(N);
	    for (size_t i = 0; i < rho.size(); i++)
	    {
		Skdiff = std::real(std::abs(rho[i]));
		Skdiff = Skdiff*Skdiff/N - Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
		phi += V[i]*U*Skdiff2;
		factor = Ifactor*V[i]*Skdiff*rho[i]*(U+Skdiff2*dU);
		if(Sk0[i] != 0)
                {
                    factor /= Sk0[i];
                }
                fy[i] = std::complex<double>(Kvec[int(i/Nk)])*factor;
		fx[i] = std::complex<double>(Kvec[int(i%Nk)])*factor;
	    }

            // Calculate Gradient
	    finufft2d2(N, &x[0], &y[0], &cx[0], -1, 1e-6, Nk, Nk, &fx[0], NULL);
	    finufft2d2(N, &x[0], &y[0], &cy[0], -1, 1e-6, Nk, Nk, &fy[0], NULL);
            for (size_t j = 0; j < N; j++)
            {
                grad[2*j]   = std::real(cx[j]);
                grad[2*j+1] = std::real(cy[j]);
            }
            return phi;
        }
        double get_energy_gradient_3d(std::vector<double>& points, std::vector<double>& grad)
        {
            std::vector<double> x(N); // x coordinate
            std::vector<double> y(N); // y coordinate
            std::vector<double> z(N); // z coordinate
	    std::vector<std::complex<double>> rho(Kmag.size());
            double Skdiff, Skdiff2, U, dU;
	    std::complex<double> factor;
            double phi = 0.0;
	    std::complex<double> Ifactor(0.0,-4.0/N);
	    grad.assign(grad.size(),0);
	    
	    int Nk = Kvec.size();
            int Nk2 = Nk*Nk;
	    for (size_t j = 0; j < N; j++)
            {
		x[j] = (points[3*j]-round(points[3*j]/L[0])+L[0])*2*M_PI/L[0];
		y[j] = (points[3*j+1]-round(points[3*j+1]/L[1])+L[1])*2*M_PI/L[1];
		z[j] = (points[3*j+2]-round(points[3*j+2]/L[2])+L[2])*2*M_PI/L[2];
	    }
            
	    finufft3d1(N, &x[0], &y[0], &z[0], &c[0], +1, 1e-6, Nk, Nk, Nk, &rho[0], NULL);
            
	    std::vector<std::complex<double>> fx(rho.size()), fy(rho.size()), fz(rho.size()), cx(N), cy(N), cz(N);
	    for (size_t i = 0; i < rho.size(); i++)
	    {
		Skdiff = std::real(std::abs(rho[i]));
		Skdiff = Skdiff*Skdiff/N - Sk0[i];
                if(Sk0[i] != 0)
                {
                    Skdiff /= Sk0[i];
                }
                Skdiff2 = Skdiff*Skdiff;
                calculate_U(U, dU, Skdiff2, Kmag[i]);
		phi += V[i]*U*Skdiff2;
		factor = Ifactor*V[i]*Skdiff*rho[i]*(U+Skdiff2*dU);
		if(Sk0[i] != 0)
                {
                    factor /= Sk0[i];
                }
                fz[i] = std::complex<double>(Kvec[int(i/Nk2)])*factor;
		fy[i] = std::complex<double>(Kvec[int((i%Nk2)/Nk)])*factor;
		fx[i] = std::complex<double>(Kvec[int(i%Nk)])*factor;
	    }

	    finufft3d2(N, &x[0], &y[0], &z[0], &cx[0], -1, 1e-6, Nk, Nk, Nk, &fx[0], NULL);
	    finufft3d2(N, &x[0], &y[0], &z[0], &cy[0], -1, 1e-6, Nk, Nk, Nk, &fy[0], NULL);
	    finufft3d2(N, &x[0], &y[0], &z[0], &cz[0], -1, 1e-6, Nk, Nk, Nk, &fz[0], NULL);
            // Calculate Gradient
            for (size_t j = 0; j < N; j++)
            {
                grad[3*j]   = std::real(cx[j]);
                grad[3*j+1] = std::real(cy[j]);
                grad[3*j+2] = std::real(cz[j]);
            }
            return phi;
        }

};

}
