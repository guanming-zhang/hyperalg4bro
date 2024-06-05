#ifndef HYPERALG_BRO_HPP
#define HYPERALG_BRO_HPP

#include <string>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <math.h>
#include "hyperalg/simple_pairwise_potential.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/distance.hpp"
#include "hyperalg/base_interaction.hpp"
#include "hyperalg/batch_cell_list_potential.hpp"
#include <vector>
#include <random>

namespace ha{

class ZeroInteraction : public ha::BaseInteraction{
public:
    ZeroInteraction(){}
    /* no definition for energy in BRO*/
    virtual double energy(double r2, const double radius_sum) const
    {
        return 0.0;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    virtual double energy_gradient(double r2, double *gij, const double radius_sum)
    {
        *gij = 0.0;
        return 0.0;
    }

    /* no definition for hessian in BRO*/
    virtual double energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) 
    {
        *gij = 0.0; 
        *hij = 0.0;
        return 0.0;
    }
};


template <size_t ndim>
class BroPeriodicCellLists : public CellListPotential< ha::ZeroInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    BroPeriodicCellLists(double mean, double sigma,
        std::vector<double> const radii, std::vector<double> const boxv,
        int noise_type=0,const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPotential< ha::ZeroInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::ZeroInteraction>(),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv)
    {
        //0--nonreciprocal uniform, 1--reciprocal uniform 
        //2--nonreciprocal gaussian, 3--reciprocal gaussian, negative number --- no noise
        //reciprocal: the noisy force on a pair of particles is oppsite and of the same magnitude
        //nonreciprocal: the noisy force on a pair of particles is oppsite and not necessarily of the same
        this->set_gradient_noise(mean,sigma,noise_type); 
    }
};

template <size_t ndim>
class BroPeriodicBatchCellLists : public BatchCellListPotential< ha::ZeroInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    BroPeriodicBatchCellLists(double mean, double sigma,
        std::vector<double> const radii, std::vector<double> const boxv,
        int noise_type=0, const double ncellx_scale=1.0, const bool balance_omp=true)
        : BatchCellListPotential<ha::ZeroInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::ZeroInteraction>(),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv)
    {
        //0--nonreciprocal uniform, 1--reciprocal uniform 
        //2--nonreciprocal gaussian, 3--reciprocal gaussian, negative number --- no noise
        //reciprocal: the noisy force on a pair of particles is oppsite and of the same magnitude
        //nonreciprocal: the noisy force on a pair of particles is oppsite and not necessarily of the same
        this->set_gradient_noise(mean,sigma,noise_type); 
    }
};




}

#endif
