#ifndef _HYPERALG_BATCH_CELL_LIST_POTENTIAL_HPP
#define _HYPERALG_BATCH_CELL_LIST_POTENTIAL_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <random>
#include <iterator>

#include "hyperalg/distance.hpp"
#include "hyperalg/cell_lists.hpp"
#include "hyperalg/vecN.hpp"
#include "hyperalg/cell_list_potential.hpp"

namespace ha{

/**
 * class which count the number of atom pairs 
 */
template <typename pairwise_interaction, typename distance_policy>
class PairCountingAccumulator{
/* protected members*/
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const std::vector<double>* m_coords;
    const std::vector<double> m_radii; 
public:
    std::vector<unsigned long> m_pair_count;     /* an array counting the number of neighbouring atom pairs for each thread(subdomain)*/
    PairCountingAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : m_interaction(interaction),
          m_dist(dist),
          m_radii(radii)
    {
        #ifdef _OPENMP
        m_pair_count.resize(omp_get_max_threads());
        #else
        m_pair_count.resize(1);
        #endif
    }
    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom){
        std::vector<double> dr(m_ndim);
        const size_t xi_off = m_ndim * atom_i;
        const size_t xj_off = m_ndim * atom_j;
        m_dist->get_rij(dr.data(), this->m_coords->data() + xi_off, this->m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        const double radius_sum = m_radii[atom_i] + m_radii[atom_j];
        if (sqrt(r2) < radius_sum ) {
            #ifdef _OPENMP
                m_pair_count[isubdom]++ ;
            #else
                m_pair_count[0]++ ;
            #endif
        }
    }
    
    void reset_data(const std::vector<double>* coords){
        m_coords = coords;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            m_pair_count[omp_get_thread_num()] = 0;
        }
        #else
            m_pair_count[0] = 0;
        #endif
    }

    unsigned long get_num_pairs(){
        unsigned long pair_count = 0;
        for(size_t i = 0; i < m_pair_count.size(); ++i) {
            pair_count += m_pair_count[i];
        }
        return pair_count;
        
    } 
};



/**
 * class which accumulates the batch energy one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class BatchEnergyAccumulator : public ha::EnergyAccumulator<pairwise_interaction, distance_policy> {
/** protected members*/
protected:
    const std::vector<unsigned long>* m_acc_pairs; /* to accumulate the number of pairs in subdomain*/
    std::vector<unsigned long> m_nvisited_pairs; /* a vector counting the number of atom pairs visited by each thread(subdomain)*/
    const std::set<unsigned long>* m_ibatch_pairs; /* a set(ordered) storing a batch of pair indices*/
public:
    /** a vector for the batch of pairs m_batch_pairs[isubdom] = a set of pairs such as {{atom1,atom2},{atom2,atom5} ...}*/
    std::vector<std::set<std::vector<unsigned long> > > m_batch_pairs; 
    /* the destructor of its base(parent) class is called by default*/
    ~BatchEnergyAccumulator(){} 
    BatchEnergyAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : EnergyAccumulator<pairwise_interaction, distance_policy> (interaction,dist,radii){
    
        #ifdef _OPENMP
            m_nvisited_pairs.resize(omp_get_max_threads(),0);
            m_batch_pairs.resize(omp_get_max_threads());
        #else
            m_nvisited_paris.resize(1,0);
            m_batch_pairs.resize(1);
        #endif 
    }

    virtual void reset_data(const std::vector<double>* coords, const std::set<unsigned long>* ibatch_pairs, const std::vector<unsigned long>* acc_pairs){
        m_ibatch_pairs = ibatch_pairs;
        m_acc_pairs = acc_pairs;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            m_nvisited_pairs[omp_get_thread_num()] = 0;
            m_batch_pairs[omp_get_thread_num()].clear();
        }
        #else
            m_nvisited_pairs[0] = 0;
            m_batch_pairs[0].clear();
        #endif
        EnergyAccumulator<pairwise_interaction, distance_policy>::reset_data(coords);
    }

    virtual void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        /** a check must be done first
            to make sure atom_i and atom_j are neigbhours*/
        /* The pair energy is added 
           once the ivisited_pair hits one pair index in m_batch_pairs*/
        ha::VecN<this->m_ndim, double> dr;
        const size_t xi_off = this->m_ndim * atom_i;
        const size_t xj_off = this->m_ndim * atom_j;
        
        // note that if we use a template class, we have to use this->m_dist to call the inherited member
        this->m_dist->get_rij(dr.data(), this->m_coords->data() + xi_off, this->m_coords->data() + xj_off);
        
        double r2 = 0;
        for (size_t k = 0; k < this->m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        
        double radius_sum = 0;
        
        if(this->m_radii.size() > 0) {
            radius_sum = this->m_radii[atom_i] + this->m_radii[atom_j];
        }
    
        unsigned long ivisited_pairs = m_nvisited_pairs[isubdom] + m_acc_pairs->at(isubdom);
        
        if (sqrt(r2) < radius_sum ) {
            m_nvisited_pairs[isubdom] ++;
            if (m_ibatch_pairs->count(ivisited_pairs) > 0){
                #ifdef _OPENMP
                *this->m_energies[isubdom] += this->m_interaction->energy(r2, radius_sum);
                m_batch_pairs[isubdom].insert({atom_i,atom_j});
                #else
                *this->m_energies[0] += this->m_interaction->energy(r2, radius_sum);
                m_batch_pairs[0].insert({atom_i,atom_j});
                #endif
            }
        }
        
    }
    
};

template <typename pairwise_interaction, typename distance_policy>
class BatchEnergyGradientAccumulator: public ha::EnergyGradientAccumulator<pairwise_interaction, distance_policy> {
protected:
    const std::set<unsigned long>* m_ibatch_pairs; /* a set(ordered) storing a batch of pair indices*/
    const std::vector<unsigned long>* m_acc_pairs; /* to accumulate the number of pairs in subdomain*/
    std::vector<unsigned long> m_nvisited_pairs; /* a vector counting the number of atom pairs visited by each thread(subdomain)*/
public:    
    /** a vector for the batch of pairs m_batch_pairs[isubdom] = a set of pairs such as {{atom1,atom2},{atom2,atom5} ...}*/
    std::vector<std::set<std::vector<unsigned long> > > m_batch_pairs; 
    ~BatchEnergyGradientAccumulator(){}

    BatchEnergyGradientAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : EnergyGradientAccumulator<pairwise_interaction, distance_policy>(interaction,dist,radii)
        {
        #ifdef _OPENMP
            m_nvisited_pairs.resize(omp_get_max_threads(),0);
            m_batch_pairs.resize(omp_get_max_threads());
        #else
            m_nvisited_pairs.resize(1,0);
            m_batch_pairs.resize(1);
        #endif 

    }
        
    virtual void reset_data(const std::vector<double> * coords, std::vector<double> * gradient,
                           const std::set<unsigned long>* ibatch_pairs, const std::vector<unsigned long>* acc_pairs) {
        m_ibatch_pairs = ibatch_pairs;
        m_acc_pairs = acc_pairs;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            m_nvisited_pairs[omp_get_thread_num()] = 0;
            m_batch_pairs[omp_get_thread_num()].clear();
        }
        #else
            m_nvisited_pairs[0] = 0;
            m_batch_pairs[0].clear();
        #endif
        EnergyGradientAccumulator<pairwise_interaction, distance_policy>::reset_data(coords,gradient);
    }

    virtual void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        ha::VecN<this->m_ndim, double> dr;
        const size_t xi_off = this->m_ndim * atom_i;
        const size_t xj_off = this->m_ndim * atom_j;
        this->m_dist->get_rij(dr.data(), this->m_coords->data() + xi_off, this->m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < this->m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double gij;
        double radius_sum = 0;
        if(this->m_radii.size() > 0) {
            radius_sum = this->m_radii[atom_i] + this->m_radii[atom_j];
        }
        unsigned long ivisited_pairs = m_nvisited_pairs[isubdom] + m_acc_pairs->at(isubdom);
        if (sqrt(r2) < radius_sum ) { //chech if atom i and j is a pair of neighbours
            m_nvisited_pairs[isubdom] ++;
            if (m_ibatch_pairs->count(ivisited_pairs) > 0){ //check the current pair is in the batch
                #ifdef _OPENMP
                *this->m_energies[isubdom] += this->m_interaction->energy_gradient(r2, &gij, radius_sum);
                m_batch_pairs[isubdom].insert({atom_i,atom_j});
                #else
                *this->m_energies[0] += this->m_interaction->energy_gradient(r2, &gij, radius_sum);
                m_batch_pairs[0].insert({atom_i,atom_j});
                #endif
                
                //for (size_t k = 0; k < this->m_ndim; ++k) {
                //    double dr_hat_k = dr[k]/sqrt(r2);
                //    (*this->m_gradient)[xi_off + k] -= dr_hat_k*this->random_number();
                //    (*this->m_gradient)[xj_off + k] += dr_hat_k*this->random_number();                
                //}

                if ((this->m_noise_type == 0) || (this->m_noise_type == 2)){
                    for (size_t k = 0; k < this->m_ndim; ++k) {
                        double dr_hat_k = dr[k]/sqrt(r2);
                        (*this->m_gradient)[xi_off + k] -= dr_hat_k*this->random_number();
                        (*this->m_gradient)[xj_off + k] += dr_hat_k*this->random_number();
                    }
                }
                else if ((this->m_noise_type == 1) || (this->m_noise_type == 3)){
                    double rnd = this->random_number();
                    for (size_t k = 0; k < this->m_ndim; ++k) {
                        double dr_hat_k = dr[k]/sqrt(r2);
                        (*this->m_gradient)[xi_off + k] -= dr_hat_k*rnd;
                        (*this->m_gradient)[xj_off + k] += dr_hat_k*rnd;
                    }
                }

                if (gij != 0) {
                    for (size_t k = 0; k < this->m_ndim; ++k) {
                        dr[k] *= gij;
                        (*this->m_gradient)[xi_off + k] -= dr[k];
                        (*this->m_gradient)[xj_off + k] += dr[k];
                    }
                }
            }
        }
    }

};

/**
 * class which accumulates the energy, gradient, and Hessian one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class BatchEnergyGradientHessianAccumulator: public ha::EnergyGradientHessianAccumulator<pairwise_interaction, distance_policy>{
protected:
    const std::set<unsigned long>* m_ibatch_pairs; /* a set(ordered) storing a batch of pair indices*/
    const std::vector<unsigned long>* m_acc_pairs; /* to accumulate the number of pairs in subdomain*/
    std::vector<unsigned long> m_nvisited_pairs; /* a vector counting the number of atom pairs visited by each thread(subdomain)*/
public:
    /** a vector for the batch of pairs m_batch_pairs[isubdom] = a set of pairs such as {{atom1,atom2},{atom2,atom5} ...}*/
    std::vector<std::set<std::vector<unsigned long> > > m_batch_pairs;

    ~BatchEnergyGradientHessianAccumulator(){}

    BatchEnergyGradientHessianAccumulator(std::shared_ptr<pairwise_interaction> & interaction,
            std::shared_ptr<distance_policy> & dist,
            std::vector<double> const & radii)
        : EnergyGradientHessianAccumulator<pairwise_interaction, distance_policy>(interaction,dist,radii)
        {
    
        #ifdef _OPENMP
            m_nvisited_pairs.resize(omp_get_max_threads(),0);
            m_batch_pairs.resize(omp_get_max_threads());
        #else
            m_nvisited_pairs.resize(1,0);
            m_batch_pairs.resize(1);
        #endif
    }
    virtual void reset_data(const std::vector<double> * coords, std::vector<double> * gradient,std::vector<double> * hessian,
                           const std::set<unsigned long>* ibatch_pairs, const std::vector<unsigned long>* acc_pairs) {
        m_ibatch_pairs = ibatch_pairs;
        m_acc_pairs = acc_pairs;
       #ifdef _OPENMP
        #pragma omp parallel
        {
            m_nvisited_pairs[omp_get_thread_num()] = 0;
            m_batch_pairs[omp_get_thread_num()].clear();
        }
        #else
            m_nvisited_pairs[0] = 0;
            m_batch_pairs[0].clear();
        #endif
        EnergyGradientHessianAccumulator<pairwise_interaction, distance_policy>::reset_data(coords,gradient,hessian);
    }

    void insert_atom_pair(const size_t atom_i, const size_t atom_j, const size_t isubdom)
    {
        ha::VecN<this->m_ndim, double> dr;
        const size_t xi_off = this->m_ndim * atom_i;
        const size_t xj_off = this->m_ndim * atom_j;
        this->m_dist->get_rij(dr.data(), this->m_coords->data() + xi_off, this->m_coords->data() + xj_off);
        double r2 = 0;
        for (size_t k = 0; k < this->m_ndim; ++k) {
            r2 += dr[k] * dr[k];
        }
        double gij, hij;
        double radius_sum = 0;
        if(this->m_radii.size() > 0) {
            radius_sum = this->m_radii[atom_i] + this->m_radii[atom_j];
        }
        unsigned long ivisited_pairs = m_nvisited_pairs[isubdom] + m_acc_pairs->at(isubdom);
        
        if (sqrt(r2) < radius_sum ) { //chech if atom i and j is a pair of neighbours
            m_nvisited_pairs[isubdom] ++;
            if (m_ibatch_pairs->count(ivisited_pairs) > 0){ //check the current pair is in the batch
                #ifdef _OPENMP
                m_batch_pairs[isubdom].insert({atom_i,atom_j});
                #else
                m_batch_pairs[0].insert({atom_i,atom_j});
                #endif
                EnergyGradientHessianAccumulator<pairwise_interaction, distance_policy>::insert_atom_pair(atom_i, atom_j, isubdom);
            }
        }
    }

};




/**
 * This class inherit CellListPotential, allowing all the functions such as get_energy(...) in CellListPotential
 * On top of the base class, pperations acting on a batch of atom pairs such as get_batch_energy(...) are implemented . 
 */

template <typename pairwise_interaction, typename distance_policy>
class BatchCellListPotential : public ha::CellListPotential<pairwise_interaction, distance_policy> {
protected:
     /** a vector for the batch of pairs m_tot_batch_pairs = a set of pairs such as {{atom1,atom2},{atom2,atom5} ...}*/
    std::set<std::vector<unsigned long> >  m_tot_batch_pairs;
    unsigned long m_npairs;
    bool m_initialized,m_store_batch_atoms;
    std::set<unsigned long > m_ibatch_pairs; /* the indices of atom pairs in the batch*/
    std::vector<unsigned long > m_acc_pairs; /* the accumulated number of pairs in subdomains*/
    std::vector<double> m_batch_atom_coords; /*record the old coords for updating the cell list*/
    std::vector<long> m_ibatch_atoms; /*record the atom indicices in the batch*/
    /** accumulators for 1 batch energy 2 batch gradient 3 batch Hessian*/
    BatchEnergyAccumulator<pairwise_interaction,distance_policy> m_beAcc; 
    BatchEnergyGradientAccumulator<pairwise_interaction,distance_policy> m_begAcc; 
    BatchEnergyGradientHessianAccumulator<pairwise_interaction, distance_policy> m_beghAcc;
    PairCountingAccumulator<pairwise_interaction, distance_policy> m_pAcc;

public:
     
     BatchCellListPotential(
            std::shared_ptr<pairwise_interaction> interaction,
            std::shared_ptr<distance_policy> dist,
            std::vector<double> const & boxvec,
            double rcut, double ncellx_scale,
            const std::vector<double> radii,
            const bool balance_omp=true)
        :CellListPotential<pairwise_interaction, distance_policy>(interaction, dist, boxvec,rcut, ncellx_scale,radii,balance_omp),
          m_beAcc(interaction, dist, radii),
          m_begAcc(interaction, dist, radii),
          m_beghAcc(interaction, dist, radii),
          m_pAcc(interaction, dist, radii),
          m_npairs(0),
          m_initialized(false),
          m_store_batch_atoms(false),
          m_gen(std::random_device{}()){
            #if _OPENMP
                m_acc_pairs.resize(omp_get_max_threads(),0);
            #else
                m_acc_pairs.resize(1,0);
            #endif
         }

    unsigned long initialize_batch_by_size(std::vector<double> const & coords,unsigned long batch_size,bool reset_clist=true){
        const size_t natoms = coords.size() / this->m_ndim;
        if (this->m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            throw std::runtime_error("coords is not finite");
        }
        prepare_pairs(coords,reset_clist); 
        prepare_batch(batch_size);   
        return m_npairs;
    }

    unsigned long initialize_batch_by_fraction(std::vector<double> const & coords,double fraction,bool reset_clist=true){
        const size_t natoms = coords.size() / this->m_ndim;
        if (this->m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            throw std::runtime_error("coords is not finite");
        }

        if ((fraction>1.0) || (fraction<0.0)) {
            throw std::runtime_error("fraction must be in [0,1]");
        }
        prepare_pairs(coords,reset_clist);
        //m_npairs is updated in prepare_pairs(.)
        unsigned long batch_size = std::min((unsigned long)(fraction*m_npairs + 1.0), m_npairs); 
        prepare_batch(batch_size); 
        return m_npairs;
    }

    virtual double get_batch_energy(std::vector<double> const & coords)
    {
        const size_t natoms = coords.size() / this->m_ndim;
        if (this->m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return NAN;
        }

        m_beAcc.reset_data(&coords,&m_ibatch_pairs,&m_acc_pairs);
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_beAcc);
        looper.loop_through_atom_pairs();
        m_tot_batch_pairs.clear();
        for (size_t isubdom=0;isubdom<m_beAcc.m_batch_pairs.size();isubdom++){
            for (auto pair:m_beAcc.m_batch_pairs[isubdom]){
                m_tot_batch_pairs.insert(pair);
            }
        }
        //get the vector of batch atoms in the last call 
        if (m_store_batch_atoms){
            std::set<long> set_ibacth_atoms;
            for (auto pair:m_tot_batch_pairs){
                set_ibacth_atoms.insert(pair[0]);
                set_ibacth_atoms.insert(pair[1]);
            }
            m_ibatch_atoms.clear();
            for (auto iatom:set_ibacth_atoms){ 
                m_ibatch_atoms.push_back(iatom);
            }
            // strore batch coords for updating cell list 
            m_batch_atom_coords.clear();
            for (auto ibatch_atoms:m_ibatch_atoms){
                for (size_t k=0; k< this->m_ndim; k++ ){
                    m_batch_atom_coords.push_back(coords[ibatch_atoms*this->m_ndim + k]);
                }
            }
        }
        return m_beAcc.get_energy();
    }

    virtual double get_batch_energy_gradient(std::vector<double> const & coords, std::vector<double> & grad)
    {
        //this->m_cell_lists.print_container();
        const size_t natoms = coords.size() / this->m_ndim;
        if (this->m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != grad.size()) {
            throw std::invalid_argument("the gradient has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(grad.begin(), grad.end(), NAN);
            return NAN;
        }
        //CellListPotential<pairwise_interaction, distance_policy>::update_iterator(coords);
        //std::fill(grad.begin(), grad.end(), 0.);
        m_begAcc.reset_data(&coords,&grad,&m_ibatch_pairs,&m_acc_pairs);
        
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_begAcc);
        looper.loop_through_atom_pairs();
        
        m_tot_batch_pairs.clear();
        for (size_t isubdom=0;isubdom<m_begAcc.m_batch_pairs.size();isubdom++){
            for (auto pair:m_begAcc.m_batch_pairs[isubdom]){
                m_tot_batch_pairs.insert(pair);
            }
        }
        //get the vector of batch atoms in the laster call 
        if (m_store_batch_atoms){
            std::set<long> set_ibacth_atoms;
            for (auto pair:m_tot_batch_pairs){
                set_ibacth_atoms.insert(pair[0]);
                set_ibacth_atoms.insert(pair[1]);
            }
            m_ibatch_atoms.clear();
            m_batch_atom_coords.clear();
            for (auto iatom:set_ibacth_atoms){ 
                m_ibatch_atoms.push_back(iatom);
                // strore batch coords for updating cell list 
                for (size_t k=0; k< this->m_ndim; k++ ){
                    m_batch_atom_coords.push_back(coords[iatom*this->m_ndim + k]);
                }
            }
        }
        return m_begAcc.get_energy();
    }

    virtual double get_batch_energy_gradient_hessian(std::vector<double> const & coords,
            std::vector<double> & grad, std::vector<double> & hess)
    {
        const size_t natoms = coords.size() / this->m_ndim;
        if (this->m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (coords.size() != grad.size()) {
            throw std::invalid_argument("the gradient has the wrong size");
        }
        if (hess.size() != coords.size() * coords.size()) {
            throw std::invalid_argument("the Hessian has the wrong size");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            std::fill(grad.begin(), grad.end(), NAN);
            std::fill(hess.begin(), hess.end(), NAN);
            return NAN;
        }

        std::fill(grad.begin(), grad.end(), 0.);
        std::fill(hess.begin(), hess.end(), 0.);
        m_beghAcc.reset_data(&coords,&grad,&hess,&m_ibatch_pairs,&m_acc_pairs);
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_beghAcc);
        
        looper.loop_through_atom_pairs();

        m_tot_batch_pairs.clear();
        for (size_t isubdom=0;isubdom<m_beghAcc.m_batch_pairs.size();isubdom++){
            for (auto pair:m_beghAcc.m_batch_pairs[isubdom]){
                m_tot_batch_pairs.insert(pair);
            }
        }
        
        //get the vector of batch atoms in the laster call 
        if (m_store_batch_atoms){
            std::set<unsigned long> set_ibacth_atoms;
            for (auto pair:m_tot_batch_pairs){
                set_ibacth_atoms.insert(pair[0]);
                set_ibacth_atoms.insert(pair[1]);
            }
            m_ibatch_atoms.clear();
            for (auto iatom:set_ibacth_atoms){ m_ibatch_atoms.push_back(iatom);}
            // strore batch coords for updating cell list 
            m_batch_atom_coords.clear();
            for (auto ibatch_atoms:m_ibatch_atoms){
                for (size_t k=0; k< this->m_ndim; k++ ){
                    m_batch_atom_coords.push_back(coords[ibatch_atoms*this->m_ndim + k]);
                }
            }
        }
        return m_beghAcc.get_energy();
    }

    virtual void get_batch_pairs(std::set<std::vector<unsigned long>> & batch_pairs){
        // this is not very efficient, but it is correct
        batch_pairs = m_tot_batch_pairs;
    }

    virtual double get_energy(std::vector<double> const & coords){
        // the cell list is recalculated when doing the full gradient, 
        // if we want to update_specific atoms later (when reset_clist==flase)
        // we need to reinitialise the cell_list
        m_initialized = false;
        return CellListPotential<pairwise_interaction, distance_policy>::get_energy(coords);
    }

    virtual double get_energy_gradient(std::vector<double> const & coords, std::vector<double> & grad){
        m_initialized = false;
        return CellListPotential<pairwise_interaction, distance_policy>::get_energy_gradient(coords,grad);
    }

    virtual double get_energy_gradient_hessian(std::vector<double> const & coords,
            std::vector<double> & grad, std::vector<double> & hess){
        m_initialized = false;
        return CellListPotential<pairwise_interaction, distance_policy>::get_energy_gradient_hessian(coords, grad,hess);

    }
    
    virtual void get_neighbors(std::vector<double> const & coords,
                                std::vector<std::vector<size_t>> & neighbor_indss,
                                std::vector<std::vector<std::vector<double>>> & neighbor_distss,
                                const double cutoff_factor = 1.0)
    {
        m_initialized = false;
        return CellListPotential<pairwise_interaction, distance_policy>::get_neighbors( coords, neighbor_indss, neighbor_distss,
                                cutoff_factor);
    }
    
    
private:
    std::mt19937 m_gen;

    void random_shuffle(std::vector<unsigned long>& vec){
        std::shuffle(vec.begin(), vec.end(), m_gen);
    }
    void prepare_pairs(std::vector<double> const & coords,bool reset_clist){
        // update the cell list use old coords and old batch atoms in the last run
        // ,or reset the cell list
        if (reset_clist){
            this->m_cell_lists.update(coords); 
            m_store_batch_atoms = true;
        }
        else if (!m_initialized){
            this->m_cell_lists.update(coords);   
            m_initialized = true;
            m_store_batch_atoms = true;
        }
        else{
            this->m_cell_lists.update_specific(coords,m_ibatch_atoms,m_batch_atom_coords);
        }
        // count the total number of atom pairs
        m_pAcc.reset_data(&coords);
        auto looper = this->m_cell_lists.get_atom_pair_looper(m_pAcc);
        looper.loop_through_atom_pairs();
        //calculate m_acc_pairs
        std::fill(m_acc_pairs.begin(),m_acc_pairs.end(),0);
        for (size_t i=1;i<m_acc_pairs.size();i++){
            m_acc_pairs[i] = m_acc_pairs[i-1] + m_pAcc.m_pair_count[i-1];
        }
        // get a batch of pair indices
        m_npairs = m_pAcc.get_num_pairs();
    }

    void prepare_batch(unsigned long batch_size){
        if (batch_size > m_npairs) {batch_size = m_npairs;}
        std::vector<unsigned long> ibatch_pairs(m_npairs);
        for (size_t i=0;i<ibatch_pairs.size();i++){ ibatch_pairs[i] = i;}
        random_shuffle(ibatch_pairs);
        m_tot_batch_pairs.clear();
        m_ibatch_pairs.clear();
        for (size_t i=0;i<batch_size;i++){
            m_ibatch_pairs.insert(ibatch_pairs[i]);
        }
    }

};

} // end of namespace ha


#endif //#ifndef _HYPERALG_BATCH_CELL_LIST_POTENTIAL_HPP