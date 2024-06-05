#ifndef HYPERALG_BASE_INTERACTION_HPP
#define HYPERALG_BASE_INTERACTION_HPP

#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>

namespace ha{
    
    class BaseInteraction {
    public:
        virtual ~BaseInteraction(){}

        virtual double energy(const double r2, const double radius_sum) const
        {
            throw std::runtime_error("BaseInteraction::energy must be overloaded");
        }

        virtual double energy_gradient(const double r2, double *const gij, const double radius_sum) const
        {
            throw std::runtime_error("BaseInteraction::energy must be overloaded");
        }

        virtual double energy_gradient_hessian(const double r2, double *const gij,
                double *const hij, const double radius_sum) const
        {
            throw std::runtime_error("BaseInteraction::energy must be overloaded");
        }
    };
}

#endif
