// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// thorium is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// thorium is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "interpolation.h"

#include <array>
#include <vector>

namespace math
{
  namespace
  {
    typedef std::array<double, 8> polynomial_d8_t;
    double yi_li_mult(int n, double y, const float* x, int i)
    {
      for (int j = 0; j != n; ++j)
        if (i != j)
          y /= (double(x[i]) - double(x[j]));
      return y;
    }

    polynomial_d8_t li_coefs(double a)
    {
      return{
        -a,
        1,
        0,
        0,
        0,
        0,
        0,
        0
      };
    }

    // Expantion of (x-a)(x-b) = a b - a x - b x + x^2
    polynomial_d8_t li_coefs(double a, double b)
    {
      return{
        a * b,
        -a - b ,
        1,
        0,
        0,
        0,
        0,
        0
      };
    }

    // Expantion of (x-a)(x-b)(x-c)(x-d) = -a b c + a b x + a c x - a x^2 + b c x - b x^2 - c x^2 + x^3
    polynomial_d8_t li_coefs(double a, double b, double c)
    {
      return{
        -a * b * c,
        a * b + a * c + b * c,
        -a - b - c,
        1,
        0,
        0,
        0,
        0
      };
    }

    // Expantion of (x-a)(x-b)(x-c)(x-d) = a b c d - a b c x - a b d x + a b x^2 - a c d x + a c x^2 + a d x^2 - a x^3 - b c d x + b c x^2 + b d x^2 - b x^3 + c d x^2 - c x^3 - d x^3 + x^4
    polynomial_d8_t li_coefs(double a, double b, double c, double d)
    {
      return{
        a * b * c * d,
        -a * b * c - a * b * d - a * c * d - b * c * d ,
        a * b + a * c + a * d + b * c + b * d + c * d,
        -a - b - c - d,
        1,
        0,
        0,
        0
      };
    }

    polynomial_d8_t li_coefs(const std::array<double, 1>& c)
    {
      return li_coefs(c[0]);
    }

    polynomial_d8_t li_coefs(const std::array<double, 2>& c)
    {
      return li_coefs(c[0], c[1]);
    }

    polynomial_d8_t li_coefs(const std::array<double, 3>& c)
    {
      return li_coefs(c[0], c[1], c[2]);
    }

    polynomial_d8_t li_coefs(const std::array<double, 4>& c)
    {
      return li_coefs(c[0], c[1], c[2], c[3]);
    }

    template <int N>
    struct Combiner
    { 
      Combiner(const std::array<double, N>& values)
        : values_(values)
      {
        trace_.reserve(N);
      }
      
      double operator()(int k)
      {
        combination_ = 1;
        accumulator_ = 0;
        go(0, k);
        return accumulator_;
      }

    private:
      void go(int offset, int k)
      {
        if (k == 0) {
          accumulator_ += combination_;
        }
        else
        {
          for (int i = offset; i <= N - k; ++i) {
            trace_.push_back(combination_);
            combination_ *= values_[i];
            go(i + 1, k - 1);
            combination_ = trace_.back();
            trace_.pop_back();
          }
        }
      }

      const std::array<double, N>& values_;
      std::vector<double> trace_;
      double combination_;
      double accumulator_;
    };

    polynomial_d8_t li_coefs(const std::array<double, 5>& c)
    {
      Combiner<5> combine(c);
      return{
        -combine(5),
        combine(4),
        -combine(3),
        combine(2),
        -combine(1),
        1,
        0,
        0
      };
    }

    polynomial_d8_t li_coefs(const std::array<double, 6>& c)
    {
      Combiner<6> combine(c);
      return{
        combine(6),
        -combine(5),
        combine(4),
        -combine(3),
        combine(2),
        -combine(1),
        1,
        0
      };
    }

    polynomial_d8_t li_coefs(const std::array<double, 7>& c)
    {
      Combiner<7> combine(c);
      return{
        -combine(7),
        combine(6),
        -combine(5),
        combine(4),
        -combine(3),
        combine(2),
        -combine(1),
        1
      };
    }

    polynomial_d8_t& operator+=(polynomial_d8_t& l, const polynomial_d8_t& r)
    {
      for (int i = 0; i != 8; ++i)
        l[i] += r[i];
      return l;
    }

    polynomial_d8_t operator*(polynomial_d8_t l, double d)
    {
      for (int i = 0; i != 8; ++i)
        l[i] *= d;
      return l;
    }

    template <int N>
    struct make_lip
    {
      static lip_8_t apply(const float* x, const float* y)
      {
        polynomial_d8_t r = {};
        for (int i = 0; i != N; ++i)
        {
          std::array<double, N - 1> c;
          int k = 0;
          for (int j = 0; j != N; ++j)
          {
            if (i != j)
            {
              c[k++] = x[j];
            }
          }
          double d = yi_li_mult(N, y[i], x, i);
          r += li_coefs(c) * d;
        }

        return{ (float)r[0], (float)r[1], (float)r[2], (float)r[3], (float)r[4], (float)r[5], (float)r[6], (float)r[7] };
      }
    };
  }
  
  lip_8_t make_lip_8(int n, const float* x, const float* y)
  {
    switch (n)
    {
    case 8:
      return make_lip<8>::apply(x, y);
    case 7:
      return make_lip<7>::apply(x, y);
    case 6:
      return make_lip<6>::apply(x, y);
    case 5:
      return make_lip<5>::apply(x, y);
    case 4:
      return make_lip<4>::apply(x, y);
    case 3:
      return make_lip<3>::apply(x, y);
    case 2:
      return make_lip<2>::apply(x, y);
    }
  }

  lip_8_t make_lip_5(int n, const float* x, const float* y)
  {
    return make_lip_8(n, x, y);
  }

  float eval_polynomial(int n, const float* c, float x)
  {
    float r = c[0];
    float xp = x;
    for (int i = 1; i != n; ++i, xp *= x)
      r += c[i] * xp;
    return r;
  }

  float eval_lip_5(const lip_8_t& lip, float x)
  {
    return eval_polynomial(5, lip.c, x);
  }

  float eval_lip_6(const lip_8_t& lip, float x)
  {
    return eval_polynomial(6, lip.c, x);
  }

  float eval_lip_n(int n, const lip_8_t& lip, float x)
  {
    return eval_polynomial(n, lip.c, x);
  }
}
