// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
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

#include "gtest/gtest.h"
#include "ballistics/system.h"
#include "ballistics/libpredict_tle.h"
#include "math/operations.h"

using namespace testing;

static const float earth_equatorial_radius = 6.378137e6f;
static const float earth_polar_radius = 6.3568e6f;
static const float earth_perigee = 1.4709829e11f;
static const float earth_apogee = 1.52098232e11f;
static const float station_perigee = 407.5e3f;
static const float station_apogee = 416.5e3f;

class TleBallistics : public Test
{
public:
  TleBallistics()
    : System(0)
  {
  }

  ~TleBallistics()
  {
    system_free(System);
  }

  void Create(int attitudeType)
  {
    libpredict_tle::params_t params;
    params.clock = { &fixed_clock, &Time };
    params.orbit.attitude.type = attitudeType;
    params.orbit.tle.header = "ISS (ZARYA)";
    params.orbit.tle.line_1 = "1 25544U 98067A   16281.78259727  .00005772  00000-0  94529-4 0  9999";
    params.orbit.tle.line_2 = "2 25544  51.6441 216.4056 0006619  52.6341  35.2822 15.54077957 22501";
    System = ballistics::system_create(BALLISTICS_TLE, &params);
  }

  ballistics::prediction_t Predict(time_t time)
  {
    Time = time;

    ballistics::prediction_t prediction;
    ballistics::predict_orbit(System, &prediction);
    return prediction;
  }

  static time_t fixed_clock(void* param)
  {
    return *static_cast<time_t*>(param);
  }

  ballistics::system_t* System;
  time_t Time;
};

TEST_F(TleBallistics, EarthPositionAtNadirAttitude)
{
  Create(ATTITUDE_TYPE_NADIR);
  time_t t = time(0);
  ballistics::prediction_t prediction = Predict(t);
  ASSERT_NEAR(0, prediction.earth_position.x, 5);
  ASSERT_NEAR(0, prediction.earth_position.y, 5);

  ASSERT_LE(prediction.earth_position.z - earth_equatorial_radius, station_apogee);
  ASSERT_GE(prediction.earth_position.z - earth_polar_radius, station_perigee);
}

TEST_F(TleBallistics, SunPositionAtSunAttitude)
{
  Create(ATTITUDE_TYPE_SUN);
  time_t t = 1475902661; // Equinox at 2016/03/20 04:30
  ballistics::prediction_t prediction = Predict(t);
  
  ASSERT_NEAR(0, prediction.sun_position.x, station_perigee);
  ASSERT_NEAR(0, prediction.sun_position.y, station_perigee);
  
  ASSERT_LE(prediction.sun_position.z, earth_apogee + station_perigee);
  ASSERT_GE(prediction.sun_position.z, earth_perigee - station_perigee);
}

TEST_F(TleBallistics, EarthPositionAtOrbitalAttitude)
{
  Create(ATTITUDE_TYPE_ORBIT);
  time_t t = time(0);
  ballistics::prediction_t prediction = Predict(t);

  ASSERT_LE(fabsf(prediction.earth_position.z), 10000);

  float range = math::norm(prediction.earth_position);
  ASSERT_LE(range - earth_equatorial_radius, station_apogee);
  ASSERT_GE(range - earth_polar_radius, station_perigee);
}
