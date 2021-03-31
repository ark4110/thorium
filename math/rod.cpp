#include "rod.h"

namespace math
{
  void rod_to_six_tetrahedrons(const rod_t& rod, tetrahedron_t* result)
  {
    result[0].points[0] = rod.points[0];
    result[0].points[1] = rod.points[3];
    result[0].points[2] = rod.points[2];
    result[0].points[3] = rod.points[5];

    result[1].points[0] = rod.points[3];
    result[1].points[1] = rod.points[2];
    result[1].points[2] = rod.points[4];
    result[1].points[3] = rod.points[5];

    result[2].points[0] = rod.points[3];
    result[2].points[1] = rod.points[4];
    result[2].points[2] = rod.points[6];
    result[2].points[3] = rod.points[5];

    result[3].points[0] = rod.points[0];
    result[3].points[1] = rod.points[1];
    result[3].points[2] = rod.points[5];
    result[3].points[3] = rod.points[3];

    result[4].points[0] = rod.points[3];
    result[4].points[1] = rod.points[1];
    result[4].points[2] = rod.points[7];
    result[4].points[3] = rod.points[5];

    result[5].points[0] = rod.points[3];
    result[5].points[1] = rod.points[7];
    result[5].points[2] = rod.points[6];
    result[5].points[3] = rod.points[5];
  }
}
