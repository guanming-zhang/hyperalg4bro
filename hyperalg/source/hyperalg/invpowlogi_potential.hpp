#ifndef HYPERALG_INVPOWLOGI_HPP
#define HYPERALG_INVPOWLOGI_HPP

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

class InvPowLogiInteraction : public ha::BaseInteraction{
public:
    const double m_pow; // Inverse power exponent
    const double m_eps;
    const double m_k;
    InvPowLogiInteraction(double a, double eps, double k)
    : m_pow(a),
      m_eps(eps),
      m_k(k)
    {}

    /* calculate energy from distance squared */
    /* E = eps/pow*[(1-r/r_sum)^pow + 1/(1+exp[(1-r/rs)*k] ]*/
    virtual double energy(double r2, const double radius_sum) const
    {
      double E;
      const double r = std::sqrt(r2);
      if (r >= radius_sum) {
          E = 0.;
      }
      else {
          const double factor = 1.0/(1+std::exp(m_k*(1.0-r/radius_sum))) - 1.0/(1+std::exp(m_k)); 
          E = (std::pow((1 -r/radius_sum), m_pow) + factor)*m_eps/m_pow;
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
      else {
          const double factor = std::pow((1 -r/radius_sum), m_pow) * m_eps;
          /* for the derivative of logistic function*/
          const double exp = std::exp((1.0 -r/radius_sum)*m_k);
          const double El = (1.0/(1+exp) - 1.0/(1+std::exp(m_k)))*m_eps/m_pow;
          const double fl = m_k/radius_sum*exp/(1+exp);

          E =  factor / m_pow + El;
          *gij =  - factor / ((r-radius_sum)*r) + fl/((1+exp)*r)*m_eps/m_pow;
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
          // for derivatives of the logistic function
          const double exp = std::exp((1.0 -r/radius_sum)*m_k);
          const double a = m_k/radius_sum;
          const double fl = a*exp/(1+exp);
          E =  factor / m_pow;
          *gij =  - factor * denom / r + fl/((1+exp)*r)*m_eps/m_pow;
          *hij = (m_pow-1) * factor * denom * denom + fl*a*a/(1+exp)*(2*fl-1.0);
      }
      return E;
    }
};

template<size_t ndim>
class InvPowLogiCartesian : public ha::SimplePairwisePotential<ha::InvPowLogiInteraction, ha::cartesian_distance<ndim>>{
    public:
    InvPowLogiCartesian(double a, double eps, double k, const std::vector<double> radii)
    : SimplePairwisePotential<ha::InvPowLogiInteraction, ha::cartesian_distance<ndim>>
    (std::make_shared<ha::InvPowLogiInteraction>(a, eps, k),
    std::make_shared<ha::cartesian_distance<ndim>>(),
    radii)
    {}
};

template<size_t ndim>
class InvPowLogiPeriodic : public ha::SimplePairwisePotential<ha::InvPowLogiInteraction, ha::periodic_distance<ndim>>{
    public:
    const std::vector<double> m_boxv;
    InvPowLogiPeriodic(double a, double eps, double k, const std::vector<double> radii, const std::vector<double> boxv)
    : SimplePairwisePotential<ha::InvPowLogiInteraction, ha::periodic_distance<ndim>>
    (std::make_shared<ha::InvPowLogiInteraction>(a, eps, k),
    std::make_shared<ha::periodic_distance<ndim>>(boxv),
    radii),
    m_boxv(boxv)
    {}
};

template <size_t ndim>
class InvPowLogiPeriodicCellLists : public CellListPotential< ha::InvPowLogiInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    InvPowLogiPeriodicCellLists(double pow, double eps, double k,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true)
        : CellListPotential< ha::InvPowLogiInteraction, ha::periodic_distance<ndim> >
        (std::make_shared<ha::InvPowLogiInteraction>(pow, eps, k),
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
class InvPowLogiPeriodicBatchCellLists : public BatchCellListPotential< ha::InvPowLogiInteraction, ha::periodic_distance<ndim> > {
public:
    const std::vector<double> m_boxv;
    InvPowLogiPeriodicBatchCellLists(double pow, double eps, double k,
        std::vector<double> const radii, std::vector<double> const boxv,
        const double ncellx_scale=1.0, const bool balance_omp=true,double rs = 1.0)
        : BatchCellListPotential<ha::InvPowLogiInteraction, ha::periodic_distance<ndim> >(std::make_shared<ha::InvPowLogiInteraction>(pow, eps, k),
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
