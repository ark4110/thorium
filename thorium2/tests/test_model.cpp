#include "reference.h"
#include "thorium2/model.h"

#include <gtest/gtest.h>

using namespace th;
using ref::card;
using ref::num;

namespace {
const std::string kMat = card("MAT1", {"1", "7.+10", "", ".3", "2700."});

std::string top_basic() {
  return ref::top_header(10.0, 1.0, 100, 300.0) + card("MEDIUM1", {"10", "1.", "0.", "110.", "900."}) + ",1,1,1,1,1,1\n" +
         card("MEDIUM1", {"20", "1.", "0.", "200.", "800."}) + ",1,1,1,1,1,1\n";
}
}  // namespace

TEST(Model, ShellMassAreaAndFaces) {
  std::string bdf = ref::plate(1, 1, 2.0, 1.0, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".005", "", "", "", "", "0.5"}) + kMat;
  Model m = ref::model_from_text(bdf, top_basic());
  ASSERT_EQ(m.elements.size(), 1u);
  const Element& e = m.elements[0];
  EXPECT_EQ(e.dim, 2);
  EXPECT_DOUBLE_EQ(e.area, 2.0);
  EXPECT_DOUBLE_EQ(e.thickness, 0.005);
  EXPECT_DOUBLE_EQ(e.volume, 0.01);
  EXPECT_DOUBLE_EQ(e.mass, 0.01 * 2700.0 + 0.5 * 2.0);
  ASSERT_EQ(e.faces.size(), 2u);
  const Face& front = m.faces[static_cast<size_t>(e.faces[0])];
  const Face& back = m.faces[static_cast<size_t>(e.faces[1])];
  EXPECT_DOUBLE_EQ(front.normal.z, 1.0);
  EXPECT_DOUBLE_EQ(back.normal.z, -1.0);
  EXPECT_TRUE(front.radiating && back.radiating);
  EXPECT_DOUBLE_EQ(m.initial_temperature[0], 300.0);
  EXPECT_DOUBLE_EQ(m.integration.dt, 1.0);
  EXPECT_DOUBLE_EQ(m.integration.t_end, 10.0);
}

TEST(Model, SolidFacesOutwardAndInternalFacesDoNotRadiate) {
  std::string bdf = ref::hex_rod(2, 2.0, 1.0, 20) + card("PSOLID", {"20", "1"}) + kMat;
  Model m = ref::model_from_text(bdf, top_basic());
  ASSERT_EQ(m.elements.size(), 2u);
  EXPECT_NEAR(m.elements[0].volume, 1.0, 1e-12);
  EXPECT_NEAR(m.elements[0].mass, 2700.0, 1e-9);
  EXPECT_NEAR(m.elements[1].centroid.x, 1.5, 1e-12);
  int radiating = 0;
  for (const Face& f : m.faces) {
    radiating += f.radiating;
    const Element& e = m.elements[static_cast<size_t>(f.element)];
    EXPECT_GT(dot(f.normal, f.centroid - e.centroid), 0.0) << "face must point outward";
  }
  EXPECT_EQ(radiating, 10);  // 12 faces, the 2 shared ones do not radiate
  ASSERT_EQ(m.junctions.size(), 1u);
}

TEST(Model, TemperaturesFromTInitAndTSet) {
  std::string bdf = ref::plate(3, 1, 3.0, 1.0, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".005"}) + kMat;
  std::string top = top_basic() + card("T_INIT", {"1"}) + ",2,250.,3,260.\n" + card("T_SET", {"1"}) + ",3,280.\n";
  Model m = ref::model_from_text(bdf, top);
  EXPECT_DOUBLE_EQ(m.initial_temperature[0], 300.0);
  EXPECT_DOUBLE_EQ(m.initial_temperature[1], 250.0);
  EXPECT_DOUBLE_EQ(m.initial_temperature[2], 280.0);  // T_SET wins over T_INIT
  ASSERT_EQ(m.fixed.size(), 1u);
  EXPECT_EQ(m.fixed[0].element, 2);
}

TEST(Model, HeatAndRadiationSources) {
  std::string bdf = ref::plate(2, 1, 2.0, 1.0, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".005"}) + kMat;
  std::string top = top_basic() + card("HSOURCE1", {"1", "25."}) + ",1,2\n" +
                    card("RSOURCE1", {"1", "0.", "0.", "1.E11", "3.85E26", "5788.", "", "500"});
  Model m = ref::model_from_text(bdf, top);
  ASSERT_EQ(m.heat_sources.size(), 1u);
  EXPECT_EQ(m.heat_sources[0].elements, (std::vector<int>{0, 1}));
  ASSERT_EQ(m.sources.size(), 1u);
  EXPECT_DOUBLE_EQ(m.sources[0].position_at(0).z, 1e11);
  EXPECT_EQ(m.sources[0].n_photons, 500);
}

TEST(Model, ErrorsAreReported) {
  const std::string shell = ref::plate(1, 1, 1.0, 1.0, 0.0, 10, 0, 0) + kMat;
  const std::string pshell = card("PSHELL", {"10", "1", ".005"});
  // Missing MEDIUM1 for the property.
  EXPECT_THROW(ref::model_from_text(shell + card("PSHELL", {"10", "1", ".005"}),
                                    ref::top_header(1, 1, 10, 300) + card("MEDIUM1", {"99", "1.", "0.", "1.", "1."}) + ",1,1\n"),
               InputError);
  // Missing property.
  EXPECT_THROW(ref::model_from_text(shell, top_basic()), InputError);
  // Shell element with PSOLID.
  EXPECT_THROW(ref::model_from_text(shell + card("PSOLID", {"10", "1"}), top_basic()), InputError);
  // FTOP probabilities do not sum to 1.
  EXPECT_THROW(ref::model_from_text(shell + pshell,
                                    card("FTOP", {"1", "0.5", "0.", "0.6", "0."}) + top_basic().substr(top_basic().find("I_PARAM"))),
               InputError);
  // Unknown element in HSOURCE1.
  EXPECT_THROW(ref::model_from_text(shell + pshell, top_basic() + card("HSOURCE1", {"1", "1."}) + ",77\n"), InputError);
  // No initial temperature for some element.
  {
    std::string top = top_basic();
    top.erase(top.find("T_ALL"));
    EXPECT_THROW(ref::model_from_text(shell + pshell, top), InputError);
  }
  // Grid that does not exist.
  EXPECT_THROW(ref::model_from_text(card("CTRIA3", {"1", "10", "1", "2", "3"}) + pshell + kMat, top_basic()), InputError);
  // Missing I_PARAM.
  {
    std::string top = top_basic();
    const size_t a = top.find("I_PARAM");
    top.erase(a, top.find('\n', a) - a + 1);
    EXPECT_THROW(ref::model_from_text(shell + pshell, top), InputError);
  }
  // Heat capacity depending on time is not physical in the enthalpy form.
  EXPECT_THROW(ref::model_from_text(shell + pshell, ref::top_header(1, 1, 10, 300) +
                                                        card("MEDIUM1", {"10", "1.", "0.", "1.", "5"}) + ",1,1\n" +
                                                        card("DEPEND", {"5", "2."}) + ",0.,0.\n,1.\n,0.,10.\n,2.\n"),
               InputError);
  // Transmission through a solid face.
  EXPECT_THROW(ref::model_from_text(ref::hex_rod(1, 1.0, 1.0, 20) + card("PSOLID", {"20", "1"}) + kMat,
                                    card("FTOP", {"1", "0.", "0.", "0.5", "0.5", "0.5"}) +
                                        card("I_PARAM", {"BDF2", "0.", "1.", "1.", "10"}) + card("T_ALL", {"1", "300."}) +
                                        card("MEDIUM1", {"20", "1.", "0.", "1.", "1."}) + ",1,1,1,1,1,1\n"),
               InputError);
}

TEST(Model, LoadsV1Examples) {
  for (const char* name : {"Distorsion_test", "Moon_rover", "Space_telescope_1", "Space_telescope_2"}) {
    const std::string base = std::string(THORIUM2_TEST_DATA) + "/v1_examples/" + name + "/" + name;
    Model m = load_model(base + ".bdf");
    EXPECT_GT(m.elements.size(), 100u) << name;
    double mass = 0;
    for (const Element& e : m.elements) mass += e.mass;
    EXPECT_GT(mass, 0.0) << name;
    EXPECT_FALSE(m.junctions.empty()) << name;
  }
}
