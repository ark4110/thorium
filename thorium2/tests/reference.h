// Test helpers: model generators and analytic reference solutions.
#pragma once

#include "thorium2/model.h"

#include <functional>
#include <string>
#include <vector>

namespace ref {

constexpr double kSigma = 5.670374419e-8;

/// Builds a model from *.bdf and *.top texts (free or fixed field).
th::Model model_from_text(const std::string& bdf, const std::string& top);

/// Free-field card from values ("GRID,1,,0.,1.,2.").
std::string card(const std::string& name, const std::vector<std::string>& fields);
std::string num(double v);  ///< Real literal that is never mistaken for an integer ("3." not "3").

/// Rectangular plate of nx*ny CQUAD4 in the plane z = z0 spanning [0,lx]x[0,ly].
/// Grid ids start at grid0+1, element ids at elem0+1, row by row (x fastest).
/// `flip` reverses the corner order (normal -z).
std::string plate(int nx, int ny, double lx, double ly, double z0, int pid, int grid0, int elem0, bool flip = false);

/// Rod along x of n CHEXA elements with square cross-section `side`, length `len`.
std::string hex_rod(int n, double len, double side, int pid, int grid0 = 0, int elem0 = 0);

/// Common *.top text: black FTOP 1 (A = eps = 1), I_PARAM BDF2 with given time span.
std::string top_header(double t_length, double dt, int n_photons, double t_all);

// --- analytic references ---

/// Configuration factor between two directly opposed parallel rectangles a x b at distance c
/// (Howell's catalog C-11, dissertation eq. 3.2 for squares).
double F_parallel_rectangles(double a, double b, double c);
/// Configuration factor from a rectangle (width w along the common edge... see C-14) to a
/// perpendicular rectangle sharing an edge of length l: rectangle 1 has the other side h,
/// rectangle 2 has the other side w (dissertation eq. 3.1: F(H, W) with H = h/l, W = w/l, from
/// rectangle w to rectangle h).
double F_perpendicular_rectangles(double h, double w, double l);

/// Classic RK4 for y' = f(t, y) with small fixed steps; returns y(t_end).
std::vector<double> rk4(const std::function<std::vector<double>(double, const std::vector<double>&)>& f,
                        std::vector<double> y, double t0, double t_end, int steps);

}  // namespace ref
