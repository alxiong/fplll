/* Copyright (C) 2005-2008 Damien Stehle.
   Copyright (C) 2007 David Cade.
   Copyright (C) 2011 Xavier Pujol.

   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.

   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#ifndef FPLLL_LLL_H
#define FPLLL_LLL_H

#include "gso.h"
#include "gso_interface.h"
#include <functional>

FPLLL_BEGIN_NAMESPACE

/* The precision of FT must be defined before creating an instance of
   LLLReduction. The matrix b can be modified between each call to lll. */

template <class ZT, class FT> class LLLReduction
{
public:
  /**
   * Constructor.
   * The precision of FT must be defined before creating an instance of the
   * class and must remain the same until the object is destroyed (or no longer
   * needed).
   */
  LLLReduction(MatGSOInterface<ZT, FT> &m, double delta, double eta, int flags);
#ifdef FPLLL_WITH_LONG_DOUBLE
  ~LLLReduction() { LDConvHelper::free(); }
#endif
  /**
     @brief LLL reduction.

     @param kappa_min minimal index to go back to
     @param kappa_start index to start processing at
     @param kappa_end end index (exclusive)
     @param size_reduction_start only perform size reductions using vectors starting at this index
     @param pred a predicate to test partial-reduced b for early return when evaluated to true
     @return success or failure (due to numerical instability)
  */

  bool lll(int kappa_min = 0, int kappa_start = 0, int kappa_end = -1,
           int size_reduction_start = 0,
           const std::function<bool(MatGSOInterface<ZT, FT> *)> &pred = [](MatGSOInterface<ZT, FT>*){ return false; });

  /**
     @brief Single-step/iteration LLL, initialization.
     @return success or failure (due to numerical instability)
  */
  bool lll_iter_init(int kappa_min = 0, int kappa_start = 0, int kappa_end = -1, int size_reduction_start = 0);
  /**
     @brief Single-step/iteration LLL, take `step_size` number of iterations/steps.
     @return success or failure (due to numerical instability)
  */
  bool lll_iter_next(int step_size=1);

  /**
     @brief Size reduction.

     Perform size reduction for all vectors between `kappa_start` and `kappa_end`.

     @param kappa_min start index
     @param kappa_end end index (exclusive)
     @param size_reduction_start only perform size reductions using vectors starting at this index
     @return success or failure (due to numerical instability)
  */

  inline bool size_reduction(int kappa_min = 0, int kappa_end = -1, int size_reduction_start = 0);

  int status;
  int final_kappa;
  int last_early_red;
  int zeros;
  int n_swaps;

  // make public for `lll_iter_*()` usage
  MatGSOInterface<ZT, FT> &m;
private:
  /**
     @brief Size reduction.

     @param kappa index to size reduce
     @param size_reduction_end only perform size reductions using vectors up to this index
     @param size_reduction_start only perform size reductions using vectors starting at this index
     @return
  */

  bool babai(int kappa, int size_reduction_end, int size_reduction_start = 0);
  inline bool early_reduction(int start, int size_reduction_start = 0);
  inline void print_params();
  inline bool set_status(int new_status);

  FT delta, eta, swap_threshold;

  bool enable_early_red;
  bool siegel;
  bool verbose;

  vector<FT> lovasz_tests;
  vector<FT> babai_mu;
  vector<long> babai_expo;
  ZT ztmp1;
  FT mu_m_ant, ftmp1;

  // temporary states for single-iter-step lll
  int tmp_size_reduction_start;
  int tmp_kappa_min, tmp_kappa_end, tmp_kappa, tmp_kappa_max, tmp_d;
  long long tmp_iter, tmp_max_iter;

};

template <class ZT, class FT>
bool is_lll_reduced(MatGSOInterface<ZT, FT> &m, double delta, double eta);

template <class ZT, class FT>
inline bool LLLReduction<ZT, FT>::size_reduction(int kappa_min, int kappa_end,
                                                 int size_reduction_start)
{
  if (kappa_end == -1)
    kappa_end = m.d;

  extend_vect(babai_mu, kappa_end);
  extend_vect(babai_expo, kappa_end);

  for (int k = kappa_min; k < kappa_end; k++)
  {
    if ((k > 0 && !babai(k, k, size_reduction_start)) || !m.update_gso_row(k))
      return false;
  }
  return set_status(RED_SUCCESS);
}

template <class ZT, class FT>
inline bool LLLReduction<ZT, FT>::early_reduction(int start, int size_reduction_start)
{
  m.lock_cols();
  if (verbose)
  {
    cerr << "Early reduction start=" << start + 1 << endl;
  }
  for (int i = start; i < m.d; i++)
  {
    if (!babai(i, start, size_reduction_start))
      return false;
  }
  m.unlock_cols();
  last_early_red = start;
  return true;
}

template <class ZT, class FT> inline void LLLReduction<ZT, FT>::print_params()
{
  cerr << "Entering LLL"
       << "\ndelta = " << delta << "\neta = " << eta << "\nprecision = " << FT::get_prec()
       << "\nexact_dot_product = " << static_cast<int>(m.enable_int_gram)
       << "\nrow_expo = " << static_cast<int>(m.enable_row_expo)
       << "\nearly_red = " << static_cast<int>(enable_early_red)
       << "\nsiegel_cond = " << static_cast<int>(siegel)
       << "\nlong_in_babai = " << static_cast<int>(m.row_op_force_long) << endl;
}

template <class ZT, class FT> inline bool LLLReduction<ZT, FT>::set_status(int new_status)
{
  status = new_status;
  if (verbose)
  {
    if (status == RED_SUCCESS)
    {
      cerr << "End of LLL: success" << endl;
    }
    else if (status == RED_EARLY_RET)
    {
      cerr << "Early return of LLL: success" << endl;
    }
    else
    {
      cerr << "End of LLL: failure: " << RED_STATUS_STR[status] << endl;
      cerr << RED_STATUS_STR[RedStatus::RED_URL_ERR] << endl;
    }
  }
  return status == RED_SUCCESS || status == RED_EARLY_RET;
}

FPLLL_END_NAMESPACE

#endif
