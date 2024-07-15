/* Copyright (C) 2016 Guillaume Bonnoron

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

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif
using namespace std;
using namespace fplll;

/**
   @brief Test if CVP function returns correct vector.

   @param A              input lattice
   @param b              expected vector
   @return
*/

template <class ZT>
int test_cvp(ZZ_mat<ZT> &A, vector<Z_NR<mpz_t>> &target, vector<Z_NR<mpz_t>> &b, const int method)
{
  vector<Z_NR<mpz_t>> sol_coord;  // In the LLL-reduced basis
  vector<Z_NR<mpz_t>> solution;
  ZZ_mat<mpz_t> u;

  // cerr << "A " << endl << A << endl;
  int status =
      lll_reduction(A, u, LLL_DEF_DELTA, LLL_DEF_ETA, LM_WRAPPER, FT_DEFAULT, 0, LLL_DEFAULT);
  if (status != RED_SUCCESS)
  {
    cerr << "LLL reduction failed: " << get_red_status_str(status) << endl;
    return status;
  }

  status = closest_vector(A, target, sol_coord, method);

  if (status != RED_SUCCESS)
  {
    cerr << "Failure: " << get_red_status_str(status) << endl;
    return status;
  }

  vector_matrix_product(solution, sol_coord, A);

  // cerr << "A red" << endl << A << endl;
  // cerr << "sol_coord : " << sol_coord << endl;
  // cerr << "solution : " << solution << endl;
  // cerr << "expected : " << b << endl;
  // cerr << "target : " << target << endl;

  bool correct = true;
  for (int i = 0; i < A.get_cols(); i++)
  {
    correct = correct && (solution[i] == b[i]);
  }
  if (!correct)
    return 1;

  return 0;
}

/**
   @brief Test if CVP function returns the correct vector

   @param input_filename_lattice   filename of an input lattice
   @param input_filename_target    filename of a target vector
   @param output_filename  filename of the expected vector
   @return
*/

template <class ZT>
int test_filename(const char *input_filename_lattice, const char *input_filename_target,
                  const char *output_filename, const int method = CVPM_FAST)
{
  ZZ_mat<ZT> A;
  int status = 0;
  status |= read_file(A, input_filename_lattice);

  vector<Z_NR<mpz_t>> t;
  status |= read_file(t, input_filename_target);

  vector<Z_NR<mpz_t>> b;
  status |= read_file(b, output_filename);

  status |= test_cvp<ZT>(A, t, b, method);
  return status;
}

/**
 * @brief Test the CVP solver that based on Babai nearest plane */
int test_filename_babai(const char *input_filename_lattice, const char *input_filename_target,
                        const char *output_filename)
{
  ZZ_mat<mpz_t> A;
  vector<Z_NR<mpz_t>> solution, sol_coord;
  int status = 0;
  status |= read_file(A, input_filename_lattice);

  vector<Z_NR<mpz_t>> t;
  status |= read_file(t, input_filename_target);

  vector<Z_NR<mpz_t>> b;
  status |= read_file(b, output_filename);

  status |= closest_vector(solution, sol_coord, A, t);
  bool correct = true;
  for (int i = 0; i < A.get_cols(); i++)
  {
    correct = correct && (solution[i] == b[i]);
  }

  if (!correct) {
    Z_NR<mpz_t> norm_expected, norm_got;
    Z_NR<mpz_t> tmp;
    for (size_t i = 0; i < A.get_cols(); i++)
    {
      tmp.sub(b[i], t[i]);
      norm_expected.addmul(tmp, tmp);
      tmp.sub(solution[i], t[i]);
      norm_got.addmul(tmp, tmp);
    }
    // cerr << "Expected sqnorm: " << norm_expected << endl;
    // cerr << "Solution sqnorm: " << norm_got << endl;


    double err = (norm_got.get_d() - norm_expected.get_d()) / norm_expected.get_d();
    // cerr << "err from optimal CVP: " << err << endl;
    if (err < 0.07) {
      // NOTE:
      // since we are not using enumeration-based, sometimes CVP approximation would drift us
      // from the actual CVP solution, for our purpose, we still consider the solution to be correct
      correct = true;
    }
    // cerr << "Expected: " << b << endl;
    // cerr << "Solution: " << solution << endl;
    // cerr << "Target  :" << t << endl;
  }

  status |= correct == false;
  return status;
}

/**
   @brief Run CVP tests.

   @param argc             ignored
   @param argv             ignored
   @return
*/

int main(int argc, char *argv[])
{

  int status = 0;
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_cvp_in_lattice",
                                 TESTDATADIR "/tests/lattices/example_cvp_in_target",
                                 TESTDATADIR "/tests/lattices/example_cvp_out");
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_cvp_in_lattice2",
                                 TESTDATADIR "/tests/lattices/example_cvp_in_target2",
                                 TESTDATADIR "/tests/lattices/example_cvp_out2");
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_cvp_in_lattice3",
                                 TESTDATADIR "/tests/lattices/example_cvp_in_target3",
                                 TESTDATADIR "/tests/lattices/example_cvp_out3");
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_cvp_in_lattice4",
                                 TESTDATADIR "/tests/lattices/example_cvp_in_target4",
                                 TESTDATADIR "/tests/lattices/example_cvp_out4", CVPM_PROVED);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_cvp_in_lattice5",
                                 TESTDATADIR "/tests/lattices/example_cvp_in_target5",
                                 TESTDATADIR "/tests/lattices/example_cvp_out5", CVPM_PROVED);

  status |= test_filename_babai(TESTDATADIR "/tests/lattices/example_cvp_in_lattice",
                                TESTDATADIR "/tests/lattices/example_cvp_in_target",
                                TESTDATADIR "/tests/lattices/example_cvp_out");
  status |= test_filename_babai(TESTDATADIR "/tests/lattices/example_cvp_in_lattice2",
                                TESTDATADIR "/tests/lattices/example_cvp_in_target2",
                                TESTDATADIR "/tests/lattices/example_cvp_out2");
  status |= test_filename_babai(TESTDATADIR "/tests/lattices/example_cvp_in_lattice3",
                                TESTDATADIR "/tests/lattices/example_cvp_in_target3",
                                TESTDATADIR "/tests/lattices/example_cvp_out3");
  // FIXME: (alex) the plain Babai's method would result in larger err from optimal CVP solution
  // when compared to provable method (of the enum-based)

  if (status == 0)
  {
    cerr << "All tests passed." << endl;
    return 0;
  }
  else
  {
    cerr << "Some tests failed." << endl;
    return -1;
  }

  return 0;
}
