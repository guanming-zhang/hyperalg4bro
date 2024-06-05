#ifndef HYPERALG_POTENTIALS_HPP
#define HYPERALG_POTENTIALS_HPP

#include <assert.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <set>

namespace ha {
/***
 * basic potential interface for native potentials
 */
class BasePotential {
public:

    virtual ~BasePotential() {}

    virtual size_t get_ndim(){
        throw std::runtime_error("BasePotential::get_ndim must be overloaded");
    }

    /**
     * Return the energy of configuration x.  This is the only function which
     * must be overloaded
     */
    virtual double get_energy(const std::vector<double> & x)
    {
        throw std::runtime_error("BasePotential::get_energy must be overloaded");
    }

    /**
     * compute the energy and gradient.
     *
     * If not overloaded it will compute the numerical gradient
     */
    virtual double get_energy_gradient(const std::vector<double> & x, std::vector<double>& grad)
    {
        double energy = get_energy(x);
        numerical_gradient(x, grad);
        return energy;
    }

    /**
     * compute the energy and gradient and Hessian.
     *
     * If not overloaded it will compute the Hessian numerically and use get_energy_gradient
     * to get the energy and gradient.
     */
    virtual double get_energy_gradient_hessian(const std::vector<double> & x, std::vector<double> & grad,
            std::vector<double> & hess)
    {
        double energy = get_energy_gradient(x, grad);
        numerical_hessian(x, hess);
        return energy;
    }

    /**
     * compute the numerical gradient
     */
    virtual void numerical_gradient(const std::vector<double> & x, std::vector<double> & grad, double eps=1e-6)
    {
        if (x.size() != grad.size()) {
            throw std::invalid_argument("grad.size() be the same as x.size()");
        }

        std::vector<double> xnew = x;
        for (size_t i=0; i<x.size(); ++i){
            xnew[i] -= eps;
            double eminus = get_energy(xnew);
            xnew[i] += 2. * eps;
            double eplus = get_energy(xnew);
            grad[i] = (eplus - eminus) / (2. * eps);
            xnew[i] = x[i];
        }
    }

    /**
     * compute the hessian.
     *
     * If not overloaded it will call get_energy_gradient_hessian
     */
    virtual void get_hessian(const std::vector<double> & x, std::vector<double> & hess)
    {
        std::vector<double> grad(x.size());
        get_energy_gradient_hessian(x, grad, hess);
    }

    /**
     * compute the numerical gradient
     */
    virtual void numerical_hessian(const std::vector<double> & x, std::vector<double> & hess, double eps=1e-6)
    {
        if (hess.size() != x.size()*x.size()) {
            throw std::invalid_argument("hess.size() be the same as x.size()*x.size()");
        }
        size_t const N = x.size();

        std::vector<double> gplus(x.size());
        std::vector<double> gminus(x.size());

        std::vector<double> xnew = x;
        for (size_t i=0; i<x.size(); ++i){
            xnew[i] -= eps;
            get_energy_gradient(xnew, gminus);
            xnew[i] += 2. * eps;
            get_energy_gradient(xnew, gplus);
            xnew[i] = x[i];

            for (size_t j=0; j<x.size(); ++j){
                hess[N*i + j] = (gplus[j] - gminus[j]) / (2.*eps);
            }
        }
    }
 
    virtual void get_neighbors(std::vector<double> const & coords,
                                std::vector<std::vector<size_t>> & neighbor_indss,
                                std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                const double cutoff_factor = 1.0)
    {
        throw std::runtime_error("BasePotential::get_neighbors isn't overloaded");
    }

    
    virtual void get_neighbors_picky(std::vector<double> const & coords,
                                      std::vector<std::vector<size_t>> & neighbor_indss,
                                      std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                      std::vector<short> const & include_atoms,
                                      const double cutoff_factor = 1.0)
    {
        throw std::runtime_error("BasePotential::get_neighbors_picky isn't overloaded");
    }

    //These functions are valid only when you implement/overload them e.g. batch_cell_list_potential.hpp
    virtual unsigned long initialize_batch_by_size(std::vector<double> const & coords,unsigned long batch_size,
                                                  bool reset_clist = true){
        throw std::runtime_error("BasePotential::initialize_batch_by_size(.) isn't overloaded");
    }

    virtual unsigned long initialize_batch_by_fraction(std::vector<double> const & coords,double fraction,
                                                       bool reset_clist = true){
        throw std::runtime_error("BasePotential::initialize_batch_by_fraction(.) isn't overloaded");
    }

    virtual double get_batch_energy(std::vector<double> const & coords){
        throw std::runtime_error("BasePotential::get_batch_energy(.) isn't overloaded");
    }

    virtual double get_batch_energy_gradient(std::vector<double> const & coords, std::vector<double> & grad){
       throw std::runtime_error("BasePotential::get_batch_energy_gradient(.) isn't overloaded");
    }

    virtual double get_batch_energy_gradient_hessian(std::vector<double> const & coords,
            std::vector<double> & grad, std::vector<double> & hess){
        throw std::runtime_error("BasePotential::get_batch_energy_gardient_hessian(.) isn't overloaded");

    }

    virtual void get_batch_pairs(std::set<std::vector<unsigned long>> & batch_pairs){
        // this is not very efficient, but it is correct
        throw std::runtime_error("BasePotential::get_batch_pairs(.) isn't overloaded");
    }
};

} // namespace ha

#endif
