#ifndef HYPERALG_INVERSEPOWER_HPP
#define HYPERALG_INVERSEPOWER_HPP

#include <string>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <math.h>
#include "hyperalg/simple_pairwise_potential.hpp"
#include "hyperalg/cell_list_potential.hpp"
#include "hyperalg/distance.hpp"
#include "hyperalg/meta_pow.hpp"
#include "hyperalg/base_interaction.hpp"
#include "hyperalg/batch_cell_list_potential.hpp"
#include <vector>

namespace ha{

class InversePowerInteraction : public ha::BaseInteraction{
public:
    const double m_pow; // Inverse power exponent
    const double m_eps;
    
    InversePowerInteraction(double a, double eps)
    : m_pow(a),
      m_eps(eps)
    {}

    /* calculate energy from distance squared */
    virtual double energy(double r2, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      if (r >= radius_sum) {
          E = 0.;
      }
      else {
          E = std::pow((1 -r/radius_sum), m_pow) * m_eps/m_pow;
      }
      return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    virtual double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      //here we should not use "if (r2 >= radius_sum*radius_sum)" in the condition
      //because it is possible that r2 is smaller than radius_sum^2 by a very small amout(~1e-14) and the else part is excuted
      //but sqrt(r2) and radius_sum are exactly the same in the machine due to the lower persition of sqrt()
      //this will cause a problem when dividing (r-radius_sum) 
      if (r >= radius_sum) {
          E = 0.;
          *gij = 0.;
      }
      else if (abs(m_pow - 1.0) < 1e-6) {
          E = (1 - r/radius_sum)*m_eps;
          *gij = m_eps/(radius_sum*r);
      }
      else {
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          E =  factor / m_pow;
          *gij =  - factor / ((r-radius_sum)*r);
      }
      return E;
    }

    virtual double energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      if (r >= radius_sum) {
          E = 0.;
          *gij = 0;
          *hij=0;
      }
      else {
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          const double denom = 1.0 / (r-radius_sum);
          E =  factor / m_pow;
          *gij =  - factor * denom / r ;
          *hij = (m_pow-1) * factor * denom * denom;
      }
      return E;
    }
};

template<size_t ndim>
class InversePowerCartesian : public ha::SimplePairwisePotential<ha::InversePowerInteraction, ha::cartesian_distance<ndim>>{
    public:
    InversePowerCartesian(double a, double eps, const std::vector<double> radii)
    : SimplePairwisePotential<ha::InversePowerInteraction, ha::cartesian_distance<ndim>>
    (std::make_shared<ha::InversePowerInteraction>(a, eps),
    std::make_shared<ha::cartesian_distance<ndim>>(),
    radii)
    {}
};

template<size_t ndim>
class InversePowerPeriodic : public ha::SimplePairwisePotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>>{
    public:
    const std::vector<double> m_boxv;
    InversePowerPeriodic(double a, double eps, const std::vector<double> radii, const std::vector<double> boxv)
    : SimplePairwisePotential<ha::InversePowerInteraction, ha::periodic_distance<ndim>>
    (std::make_shared<ha::InversePowerInteraction>(a, eps),
    std::make_shared<ha::periodic_distance<ndim>>(boxv),
    radii),
    m_boxv(boxv)
    {}
};

template <size_t ndim>
class InversePowerPeriodicCellLists : public CellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    InversePowerPeriodicCellLists(double pow, double eps,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> >
        (std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv)
    {}
};

template <size_t ndim>
class InversePowerPeriodicBatchCellLists : public BatchCellListPotential< ha::InversePowerInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    InversePowerPeriodicBatchCellLists(double pow, double eps,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : BatchCellListPotential<ha::InversePowerInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::InversePowerInteraction>(pow, eps),
        std::make_shared<ha::periodic_distance<ndim> >(boxv),
        boxv,
        2.0 * (*std::max_element(radii.begin(), radii.end())), // rcut,
        ncellx_scale,
        radii,
        balance_omp),
        m_boxv(boxv)
    {}
};




}

#endif
