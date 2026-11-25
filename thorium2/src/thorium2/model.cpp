// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/model.h"

#include "thorium2/conduction.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>

namespace th {

int Model::element(int id) const {
  auto it = element_index.find(id);
  if (it == element_index.end()) throw std::out_of_range("element " + std::to_string(id) + " is not in the model");
  return it->second;
}

void Model::bounding_sphere(Vec3& center, double& radius) const {
  Vec3 lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  Vec3 hi = -lo;
  bool any = false;
  for (const Face& f : faces) {
    if (!f.radiating) continue;
    for (int n : f.nodes) {
      const Vec3& p = nodes[static_cast<size_t>(n)];
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
      any = true;
    }
  }
  if (!any) {
    center = {};
    radius = 0;
    return;
  }
  center = 0.5 * (lo + hi);
  radius = 0;
  for (const Face& f : faces)
    if (f.radiating)
      for (int n : f.nodes) radius = std::max(radius, norm(nodes[static_cast<size_t>(n)] - center));
}

namespace {

class Builder {
public:
  Builder(const BdfModel& bdf, const TopModel& top) : bdf_(bdf), top_(top) {}

  Model build() {
    tables();
    nodes();
    optics();
    media();
    elements();
    build_junctions(m_);
    assign_optics();
    heat_sources();
    radiation_sources();
    temperatures();
    integration();
    notes();
    return std::move(m_);
  }

private:
  [[noreturn]] static void fail(const std::string& where, const std::string& msg) { throw InputError(where + ": " + msg); }

  Property prop(const TopValue& v, const std::string& where, const std::string& what) {
    Property p;
    p.constant = v.constant;
    if (v.is_dependency) {
      auto it = m_.tables.find(v.dependency);
      if (it == m_.tables.end()) fail(where, what + " refers to DEPEND " + std::to_string(v.dependency) + ", which is not defined");
      p.table = it->second.get();
    }
    return p;
  }

  void tables() {
    for (const auto& [id, d] : top_.depend) m_.tables.emplace(id, std::make_unique<Table>(d));
  }

  void nodes() {
    for (const auto& [id, g] : bdf_.grids) {
      node_index_[id] = static_cast<int>(m_.nodes.size());
      m_.nodes.push_back(g.x);
      m_.node_ids.push_back(id);
    }
  }

  void optics() {
    for (const auto& [id, f] : top_.ftop) {
      const std::string what = "FTOP " + std::to_string(id);
      Optics o;
      o.id = id;
      o.r_dif = prop(f.r_dif, f.where, what);
      o.r_ref = prop(f.r_ref, f.where, what);
      o.absorb = prop(f.absorb, f.where, what);
      o.transmit = prop(f.transmit, f.where, what);
      o.eps = prop(f.eps, f.where, what);
      optics_index_[id] = static_cast<int>(m_.optics.size());
      optics_where_.push_back(f.where);
      m_.optics.push_back(o);
    }
  }

  void media() {
    for (const auto& [id, md] : top_.media) {
      const std::string what = "MEDIUM1 " + std::to_string(id);
      Medium x;
      x.id = id;
      x.lambda = prop(md.lambda, md.where, what + " conductivity");
      x.c = prop(md.c, md.where, what + " specific heat");
      if (x.c.depends_on_time())
        fail(md.where, what + ": specific heat may depend on temperature only (the energy balance uses the enthalpy)");
      medium_index_[id] = static_cast<int>(m_.media.size());
      medium_ftop_.push_back(md.ftop);
      medium_where_.push_back(md.where);
      m_.media.push_back(x);
    }
  }

  void check_medium_values(int mi, const std::string& where) {
    const Medium& md = m_.media[static_cast<size_t>(mi)];
    if (!md.c.table && !(md.c.constant > 0))
      fail(where, "MEDIUM1 " + std::to_string(md.id) + ": specific heat must be positive for a material of elements");
    if (md.c.table)
      for (double T : md.c.table->temperature_points())
        if (!(md.c(T, 0) > 0)) fail(md.c.table->where(), "specific heat must be positive");
    if (!md.lambda.table && md.lambda.constant < 0)
      fail(where, "MEDIUM1 " + std::to_string(md.id) + ": conductivity must not be negative");
  }

  void elements() {
    std::set<int> checked_media;
    for (const BdfElement& be : bdf_.elements) {
      Element e;
      e.id = be.id;
      e.type = be.type;
      e.dim = dimension(be.type);
      e.pid = be.pid;
      e.where = be.where;
      const std::string name = std::string(to_string(be.type)) + " " + std::to_string(be.id);
      std::vector<Vec3> xyz;
      for (int g : be.grids) {
        auto it = node_index_.find(g);
        if (it == node_index_.end()) fail(be.where, name + ": grid " + std::to_string(g) + " is not defined");
        e.nodes.push_back(it->second);
        xyz.push_back(m_.nodes[static_cast<size_t>(it->second)]);
      }
      double rho = 0, nsm = 0;
      int mid = 0;
      std::string prop_where;
      if (e.dim == 2) {
        auto it = bdf_.pshell.find(be.pid);
        if (it == bdf_.pshell.end())
          fail(be.where, name + ": a shell needs PSHELL " + std::to_string(be.pid) +
                             (bdf_.psolid.count(be.pid) ? " (found PSOLID)" : " (not defined)"));
        e.thickness = it->second.thickness;
        nsm = it->second.nsm;
        mid = it->second.mid;
        prop_where = it->second.where;
      } else {
        auto it = bdf_.psolid.find(be.pid);
        if (it == bdf_.psolid.end())
          fail(be.where, name + ": a solid needs PSOLID " + std::to_string(be.pid) +
                             (bdf_.pshell.count(be.pid) ? " (found PSHELL)" : " (not defined)"));
        mid = it->second.mid;
        prop_where = it->second.where;
      }
      auto mat = bdf_.materials.find(mid);
      if (mat == bdf_.materials.end()) fail(prop_where, "material " + std::to_string(mid) + " (MAT1 or MAT4) is not defined");
      rho = mat->second.rho;
      auto med = medium_index_.find(be.pid);
      if (med == medium_index_.end())
        fail(be.where, name + ": the *.top file has no MEDIUM1 " + std::to_string(be.pid) + " for property " + std::to_string(be.pid));
      e.medium = med->second;
      if (checked_media.insert(e.medium).second) check_medium_values(e.medium, medium_where_[static_cast<size_t>(e.medium)]);

      const auto& local_faces = element_faces(be.type);
      if (e.dim == 2) {
        e.area = area(xyz);
        if (!(e.area > 0)) fail(be.where, name + ": zero area");
        const Vec3 n = normalized(area_vector(xyz));
        e.centroid = centroid(xyz);
        e.volume = e.area * e.thickness;
        e.mass = rho * e.volume + nsm * e.area;
        if (xyz.size() == 4) {
          const double w = warping(xyz);
          if (w > 0.05) {
            warped_.push_back(be.id);
            if (w > max_warp_) max_warp_ = w, max_warp_id_ = be.id;
          }
        }
        for (int k = 0; k < 2; ++k) {
          Face f;
          f.element = static_cast<int>(m_.elements.size());
          f.local = k + 1;
          for (int i : local_faces[static_cast<size_t>(k)]) f.nodes.push_back(e.nodes[static_cast<size_t>(i)]);
          f.normal = k == 0 ? n : -n;
          f.centroid = e.centroid;
          f.area = e.area;
          e.faces.push_back(static_cast<int>(m_.faces.size()));
          m_.faces.push_back(f);
        }
      } else {
        std::vector<std::vector<Vec3>> polys;
        for (const auto& lf : local_faces) {
          std::vector<Vec3> p;
          for (int i : lf) p.push_back(xyz[static_cast<size_t>(i)]);
          polys.push_back(p);
        }
        SolidProperties sp;
        try {
          sp = solid_properties(polys);
        } catch (const std::invalid_argument& ex) {
          fail(be.where, name + ": " + ex.what());
        }
        e.volume = sp.volume;
        e.centroid = sp.centroid;
        e.mass = rho * e.volume;
        for (size_t k = 0; k < local_faces.size(); ++k) {
          Face f;
          f.element = static_cast<int>(m_.elements.size());
          f.local = static_cast<int>(k) + 1;
          for (int i : local_faces[k]) f.nodes.push_back(e.nodes[static_cast<size_t>(i)]);
          std::vector<Vec3> p = polys[k];
          if (sp.flipped[k]) {
            std::reverse(f.nodes.begin(), f.nodes.end());
            std::reverse(p.begin(), p.end());
          }
          f.area = area(p);
          f.normal = normalized(area_vector(p));
          f.centroid = centroid(p);
          e.faces.push_back(static_cast<int>(m_.faces.size()));
          m_.faces.push_back(f);
        }
      }
      if (!(e.mass > 0)) fail(be.where, name + ": mass is zero (density of material " + std::to_string(mid) + " is not given?)");
      m_.element_index[e.id] = static_cast<int>(m_.elements.size());
      m_.elements.push_back(std::move(e));
    }
    if (m_.elements.empty()) throw InputError("the *.bdf file has no elements (CTRIA3, CQUAD4, CTETRA, CPENTA, CHEXA)");
  }

  void assign_optics() {
    std::set<int> used;
    for (Face& f : m_.faces) {
      if (!f.radiating) continue;
      const Element& e = m_.elements[static_cast<size_t>(f.element)];
      const int ftop = medium_ftop_[static_cast<size_t>(e.medium)][static_cast<size_t>(f.local - 1)];
      const std::string what = std::string(to_string(e.type)) + " " + std::to_string(e.id) + " face " + std::to_string(f.local);
      if (ftop <= 0)
        fail(medium_where_[static_cast<size_t>(e.medium)],
             "MEDIUM1 " + std::to_string(m_.media[static_cast<size_t>(e.medium)].id) + " gives no FTOP for face " +
                 std::to_string(f.local) + ", needed by " + what);
      auto it = optics_index_.find(ftop);
      if (it == optics_index_.end())
        fail(medium_where_[static_cast<size_t>(e.medium)], "FTOP " + std::to_string(ftop) + " (" + what + ") is not defined");
      f.optics = it->second;
      used.insert(it->second);
      if (e.dim == 3) solid_optics_.insert(it->second);
    }
    for (int oi : used) check_optics(oi);
  }

  /// Probabilities in [0,1] summing to 1 at every table point; no transmission through solids.
  void check_optics(int oi) {
    const Optics& o = m_.optics[static_cast<size_t>(oi)];
    const std::string& where = optics_where_[static_cast<size_t>(oi)];
    std::vector<double> Ts = {300.0}, ts = {top_.iparam->t_start};
    for (const Property* p : {&o.r_dif, &o.r_ref, &o.absorb, &o.transmit, &o.eps}) {
      if (!p->table) continue;
      for (double T : p->table->temperature_points()) Ts.push_back(T);
      for (double t : p->table->time_points()) ts.push_back(t);
    }
    for (double T : Ts)
      for (double t : ts) {
        const double v[5] = {o.r_dif(T, t), o.r_ref(T, t), o.absorb(T, t), o.transmit(T, t), o.eps(T, t)};
        for (double x : v)
          if (x < -1e-12 || x > 1 + 1e-12)
            fail(where, "FTOP " + std::to_string(o.id) + ": coefficients must lie in [0, 1] (T = " + std::to_string(T) + ")");
        const double sum = v[0] + v[1] + v[2] + v[3];
        if (std::abs(sum - 1.0) > 1e-3)
          fail(where, "FTOP " + std::to_string(o.id) + ": R_dif + R_ref + A + D = " + std::to_string(sum) +
                          ", must be 1 (T = " + std::to_string(T) + ", t = " + std::to_string(t) + ")");
        if (v[3] > 0 && solid_optics_.count(oi))
          fail(where, "FTOP " + std::to_string(o.id) + " has transmission D > 0 but is used on a face of a solid element; "
                      "solid elements are opaque");
      }
  }

  int element_of(int id, const std::string& where, const std::string& card) {
    auto it = m_.element_index.find(id);
    if (it == m_.element_index.end()) fail(where, card + ": element " + std::to_string(id) + " is not in the *.bdf file");
    return it->second;
  }

  void time_only(const Property& p, const std::string& where, const std::string& what) {
    if (p.depends_on_temperature()) fail(where, what + " may depend on time only");
  }

  void heat_sources() {
    for (const auto& [id, h] : top_.hsources) {
      HeatSource s;
      s.id = id;
      s.power = prop(h.power, h.where, "HSOURCE1 " + std::to_string(id));
      time_only(s.power, h.where, "HSOURCE1 power");
      std::set<int> seen;
      for (int eid : h.elements) {
        const int ei = element_of(eid, h.where, "HSOURCE1 " + std::to_string(id));
        if (!seen.insert(ei).second) fail(h.where, "HSOURCE1 " + std::to_string(id) + ": element " + std::to_string(eid) + " is listed twice");
        s.elements.push_back(ei);
      }
      m_.heat_sources.push_back(std::move(s));
    }
  }

  void radiation_sources() {
    if (!top_.iparam) throw InputError("the *.top file has no I_PARAM card");
    for (const auto& [id, r] : top_.rsources) {
      RadiationSource s;
      s.id = id;
      s.where = r.where;
      const std::string what = "RSOURCE1 " + std::to_string(id);
      if (r.position_is_dependency) {
        auto it = m_.tables.find(r.position_dependency);
        if (it == m_.tables.end()) fail(r.where, what + ": POS DEPEND " + std::to_string(r.position_dependency) + " is not defined");
        if (it->second->value_components() < 3) fail(r.where, what + ": POS DEPEND needs 3 value components (x, y, z)");
        if (it->second->depends_on_temperature()) fail(r.where, what + ": position may depend on time only");
        s.position_is_table = true;
        s.position = it->second.get();
      } else {
        s.x = prop(r.x, r.where, what);
        s.y = prop(r.y, r.where, what);
        s.z = prop(r.z, r.where, what);
        for (const Property* p : {&s.x, &s.y, &s.z}) time_only(*p, r.where, what + " position");
      }
      s.power = prop(r.power, r.where, what);
      time_only(s.power, r.where, what + " power");
      s.n_photons = r.n_photons > 0 ? r.n_photons : top_.iparam->n_photons;
      m_.sources.push_back(s);
    }
  }

  void temperatures() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    m_.initial_temperature.assign(m_.elements.size(), top_.t_all ? *top_.t_all : nan);
    for (const TopTinit& ti : top_.t_init)
      for (const auto& [eid, T] : ti.temperatures) m_.initial_temperature[static_cast<size_t>(element_of(eid, ti.where, "T_INIT"))] = T;
    std::set<int> fixed;
    const double t0 = top_.iparam->t_start;
    for (const TopTset& ts : top_.t_set)
      for (const auto& [eid, v] : ts.temperatures) {
        const int ei = element_of(eid, ts.where, "T_SET");
        if (!fixed.insert(ei).second) fail(ts.where, "T_SET: element " + std::to_string(eid) + " is fixed twice");
        FixedTemperature f;
        f.element = ei;
        f.temperature = prop(v, ts.where, "T_SET");
        time_only(f.temperature, ts.where, "T_SET temperature");
        const double T = f.temperature(0, t0);
        if (!(T > 0)) fail(ts.where, "T_SET: temperature of element " + std::to_string(eid) + " must be positive");
        m_.initial_temperature[static_cast<size_t>(ei)] = T;
        m_.fixed.push_back(f);
      }
    std::vector<int> missing;
    for (size_t i = 0; i < m_.elements.size(); ++i)
      if (std::isnan(m_.initial_temperature[i])) missing.push_back(m_.elements[i].id);
    if (!missing.empty()) {
      std::string list;
      for (size_t i = 0; i < std::min<size_t>(missing.size(), 10); ++i) list += " " + std::to_string(missing[i]);
      throw InputError("no initial temperature (T_ALL, T_INIT or T_SET) for " + std::to_string(missing.size()) +
                       " element(s):" + list + (missing.size() > 10 ? " ..." : ""));
    }
    for (const HeatSource& h : m_.heat_sources)
      for (int ei : h.elements)
        if (fixed.count(ei))
          m_.warnings.push_back("HSOURCE1 " + std::to_string(h.id) + ": element " + std::to_string(m_.elements[static_cast<size_t>(ei)].id) +
                                " has a fixed temperature (T_SET); its heat source does not change anything");
  }

  void integration() {
    const TopIparam& p = *top_.iparam;
    m_.integration.type = p.type;
    m_.integration.t_start = p.t_start;
    m_.integration.t_end = p.t_start + p.t_length;
    m_.integration.dt = p.t_step;
    m_.integration.n_photons = p.n_photons;
    if (p.type == "EULER" || p.type == "EULER_V" || p.type == "ADAMS_4")
      m_.warnings.push_back(p.where + ": I_PARAM " + p.type +
                            " (explicit integrator of version 1) is replaced by the implicit BDF2 scheme with step t_STEP");
    if (p.kt_min || p.kt_max) m_.warnings.push_back(p.where + ": I_PARAM K_T_min/K_T_max are not used by the implicit scheme");
    if (!p.smooth.empty()) m_.warnings.push_back(p.where + ": I_PARAM smoothing is not used by the implicit scheme");
  }

  void notes() {
    if (top_.g_medium) {
      auto it = top_.media.find(*top_.g_medium);
      if (it == top_.media.end()) throw InputError("G_MEDIUM " + std::to_string(*top_.g_medium) + ": MEDIUM1 is not defined");
      const TopMedium& g = it->second;
      if (g.n.is_dependency || g.k.is_dependency || g.n.constant != 1.0 || g.k.constant != 0.0)
        m_.warnings.push_back(g.where + ": the global medium is treated as vacuum (n = 1, k = 0)");
    }
    for (const std::string& w : bdf_.warnings) m_.warnings.push_back(w);
    if (!bdf_.ignored.empty()) {
      std::string list;
      for (const auto& [name, n] : bdf_.ignored) list += " " + name + "x" + std::to_string(n);
      m_.warnings.push_back("*.bdf cards not used by the thermal analysis:" + list);
    }
    for (const std::string& w : top_.warnings) m_.warnings.push_back(w);
    if (!warped_.empty()) {
      std::string list;
      for (size_t i = 0; i < std::min<size_t>(warped_.size(), 8); ++i) list += " " + std::to_string(warped_[i]);
      m_.warnings.push_back(std::to_string(warped_.size()) + " CQUAD4 element(s) are warped by more than 5% of their size" +
                            " (largest " + std::to_string(max_warp_ * 100.0) + "% in element " + std::to_string(max_warp_id_) +
                            "):" + list + (warped_.size() > 8 ? " ..." : "") +
                            "; they are taken as 4 flat triangles around the centre");
    }
  }

  const BdfModel& bdf_;
  const TopModel& top_;
  Model m_;
  std::map<int, int> node_index_, optics_index_, medium_index_;
  std::vector<std::string> optics_where_, medium_where_;
  std::vector<std::array<int, 6>> medium_ftop_;
  std::set<int> solid_optics_;
  std::vector<int> warped_;
  double max_warp_ = 0;
  int max_warp_id_ = 0;
};

}  // namespace

Model build_model(const BdfModel& bdf, const TopModel& top) {
  if (!top.iparam) throw InputError("the *.top file has no I_PARAM card");
  return Builder(bdf, top).build();
}

Model load_model(const std::string& bdf_path, const std::string& top_path) {
  std::string top = top_path;
  if (top.empty()) {
    std::filesystem::path p(bdf_path);
    p.replace_extension(".top");
    top = p.string();
  }
  if (!std::filesystem::exists(bdf_path)) throw InputError(bdf_path + ": file not found");
  if (!std::filesystem::exists(top)) throw InputError(top + ": file not found (the *.top file must have the same name as the *.bdf file, or be given explicitly)");
  return build_model(read_bdf_file(bdf_path), read_top_file(top));
}

}  // namespace th
