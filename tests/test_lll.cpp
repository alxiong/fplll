/* Copyright (C) 2015 Martin Albrecht

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

#include "test_utils.h"
#include <cstring>
#include <fplll/fplll.h>
#include <functional>

using namespace std;
using namespace fplll;

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif

/**
   @brief Test the tester.

   @param A
   @return zero on success.
*/

template <class Z> int test_test(ZZ_mat<Z> &A)
{
  ZZ_mat<Z> U;
  ZZ_mat<Z> UT;

  MatGSO<Z_NR<Z>, FP_NR<mpfr_t>> M(A, U, UT, 0);

  int is_reduced = is_lll_reduced<Z_NR<Z>, FP_NR<mpfr_t>>(M, LLL_DEF_DELTA, LLL_DEF_ETA);

  if (is_reduced)
    cerr << "is_lll_reduced reports success when it should not" << endl;

  return is_reduced;
}

/**
   @brief Test LLL reduction.

   @param A                test matrix
   @param method           LLL method to test
   @param float_type       floating point type to test
   @param flags            flags to use
   @param prec             precision used for is_lll_reduced

   @return zero on success.
*/

template <class Z>
int test_lll(ZZ_mat<Z> &A, LLLMethod method, FloatType float_type, int flags = LLL_DEFAULT,
             int prec = 0)
{

  ZZ_mat<Z> U;
  ZZ_mat<Z> UT;

  int status = 0;

  // zero on success
  if (test_test(A))
  {
    return 1;
  }

  // zero on success
  status = lll_reduction(A, LLL_DEF_DELTA, LLL_DEF_ETA, method, float_type, 0, flags);
  if (status != RED_SUCCESS)
  {
    cerr << "LLL reduction failed with error '" << get_red_status_str(status);
    cerr << "' for method " << LLL_METHOD_STR[method];
    cerr << " and float type " << FLOAT_TYPE_STR[float_type] << endl;
    return status;
  }

  const int old_prec = prec ? FP_NR<mpfr_t>::set_prec(prec) : 0;

  MatGSO<Z_NR<Z>, FP_NR<mpfr_t>> M(A, U, UT, 0);

  // one on success
  status = is_lll_reduced<Z_NR<Z>, FP_NR<mpfr_t>>(M, LLL_DEF_DELTA, LLL_DEF_ETA);

  if (prec)
    FP_NR<mpfr_t>::set_prec(old_prec);

  if (status == 0)
    cerr << "Output of LLL reduction is not LLL reduced with method " << LLL_METHOD_STR[method]
         << endl;

  return (status == 0);
}

/**
   @brief Test LLL for matrix stored in file pointed to by `input_filename`.

   @param input_filename   a path
   @param method           LLL method to test
   @param float_type       floating point type to test
   @param flags            flags to use
   @param prec             precision used for is_lll_reduced

   @return zero on success
*/

template <class Z>
int test_filename(const char *input_filename, LLLMethod method, FloatType float_type = FT_DEFAULT,
                  int flags = LLL_DEFAULT, int prec = 0)
{
  ZZ_mat<Z> A;
  int status = 0;
  status |= read_file(A, input_filename);
  status |= test_lll<Z>(A, method, float_type, flags, prec);
  return status;
}

/**
   @brief Construct d × (d+1) integer relations matrix with bit size b and test LLL.

   @param d                dimension
   @param b                bit size
   @param method           LLL method to test
   @param float_type       floating point type to test
   @param flags            flags to use
   @param prec             precision used for is_lll_reduced

   @return zero on success
*/

template <class Z>
int test_int_rel(int d, int b, LLLMethod method, FloatType float_type = FT_DEFAULT,
                 int flags = LLL_DEFAULT, int prec = 0)
{
  ZZ_mat<Z> A;
  A.resize(d, d + 1);
  A.gen_intrel(b);
  return test_lll<Z>(A, method, float_type, flags, prec);
}

int main(int /*argc*/, char ** /*argv*/)
{

  int status = 0;
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", LM_WRAPPER, FT_DEFAULT,
                                 LLL_DEFAULT, 128);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", LM_PROVED, FT_MPFR);

  status |= test_int_rel<mpz_t>(50, 1000, LM_FAST, FT_DOUBLE);
  status |= test_int_rel<mpz_t>(50, 1000, LM_PROVED, FT_MPFR);

  status |= test_int_rel<mpz_t>(30, 2000, LM_HEURISTIC, FT_DPE);
  status |= test_int_rel<mpz_t>(30, 2000, LM_PROVED, FT_DPE);
  status |= test_int_rel<mpz_t>(30, 2000, LM_PROVED, FT_MPFR);

  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", LM_HEURISTIC);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", LM_FAST, FT_DOUBLE);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", LM_PROVED, FT_MPFR);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", LM_FAST, FT_DOUBLE,
                                 LLL_DEFAULT | LLL_EARLY_RED);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", LM_HEURISTIC, FT_DEFAULT,
                                 LLL_DEFAULT | LLL_EARLY_RED);

  // test early-return predicate, which require direct invocation of LLLReduction.lll()
  // instead of via the Wrapper, as the ZT, FT both has to be known to define `pred`
  std::function<bool(MatGSOInterface<Z_NR<mpz_t>, FP_NR<mpfr_t>> *)> pred;
  pred = [](MatGSOInterface<Z_NR<mpz_t>, FP_NR<mpfr_t>> * m) {
    return m->get_rows_of_b() % 2 == 0;
  };
  int d = 50, b = 1000;
  ZZ_mat<mpz_t> A, U, UT;
  A.resize(d, d + 1);
  A.gen_intrel(b);

  MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> M(A, U, UT, 0);
  LLLReduction<Z_NR<mpz_t>, FP_NR<mpfr_t>> lll_obj(M, LLL_DEF_DELTA, LLL_DEF_ETA, LLL_DEFAULT);
  status |= lll_obj.lll(0, 0, -1, 0, pred) == 0;
  if (lll_obj.status != RED_EARLY_RET) {
    cerr << "Fail to early-return on trivial predicate" << endl;
    return -1;
  }

  // a slightly more realistic predicate
  // this is also probabilistically satisified, but most likely the B[0].norm is not that big
  std::function<bool(MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> *)> pred2 = [](MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> *m) {
    Z_NR<mpz_t> res;
    m->b[0].dot_product(res, m->b[0]);
    return res < 10000000000000;
  };
  // one complication is the dyanmic pointer casting from derived class to base class
  std::function<bool(MatGSOInterface<Z_NR<mpz_t>, FP_NR<mpfr_t>> *)> wrapped_pred2;
  wrapped_pred2 = [pred2](MatGSOInterface<Z_NR<mpz_t>, FP_NR<mpfr_t>> *m) {
    MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>>* derived_m = dynamic_cast<MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>>*>(m);
    if (derived_m != nullptr) {
      return pred2(derived_m);
    }
    return false;
  };
  status |= lll_obj.lll(0, 0, -1, 0, wrapped_pred2) == 0;
  if (lll_obj.status != RED_EARLY_RET) {
    cerr << "Fail to early-return on trivial predicate" << endl;
    return -1;
  }

  if (status == 0)
  {
    cerr << "All tests passed." << endl;
    return 0;
  }
  else
  {
    return -1;
  }
}
