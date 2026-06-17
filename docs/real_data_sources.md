# Real Data Sources

## Selected Sources

### Wearable IMU fall data

- Source: `jasonkau/fall-detection-dataset-IMU`
- URL: https://github.com/jasonkau/fall-detection-dataset-IMU
- License: MIT
- Why it fits: public inertial sensor recordings with fall events and daily activities. The transformed dataset provides real `imu_acceleration`, angular velocity, body-angle proxy, and worker-sway proxy features.
- Limitation: falls are controlled/simulated recordings, not rope-access or high-rise cleaning incidents. They are still real sensor recordings, not generated rows.

### Historical wind data

- Source: Open-Meteo Historical Weather API
- URL: https://open-meteo.com/en/docs/historical-weather-api
- Why it fits: no API key is needed, and hourly historical `wind_speed_10m` / `wind_gusts_10m` can be fetched by latitude and longitude.
- Limitation: it provides local weather conditions, not building-face wind tunnel data. For product validation, field anemometer logs near the worker are still needed.

### Korea official weather alternative

- Source: Korea Meteorological Administration ASOS hourly data
- URL: https://www.data.go.kr/data/15057210/openapi.do
- Why it fits: official Korean hourly weather observations with wind fields and public-data attribution.
- Limitation: it requires public-data portal API usage approval/key, so Open-Meteo is easier for a reproducible public repo.

## Data Not Found As Public Open Data

- Peter Anchor rope tension time series
- Harness pressure time series from rope-access workers
- Suction readiness logs from the proposed hardware
- Real high-rise exterior-cleaning near-fall events with synchronized IMU, wind, rope tension, and harness pressure

These fields should not be synthesized and presented as measured data. They should be listed as future field-collection requirements.

## Additional Sources for Wearable Robot Extension

The current dataset supports sensing and fall-risk detection, not strength assistance. To model a wearable robot or exosuit, the project needs EMG, joint kinematics, joint torque, force/load, or actuator-assistance data.

Candidate sources:

- Lower-limb biomechanics and wearable sensors dataset: https://www.nature.com/articles/s41597-023-02840-6
- Georgia Tech Camargo lower-limb biomechanics dataset: https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/
- ENABL3S bilateral lower-limb neuromechanical dataset: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00014/full
- EMG-based exoskeleton torque prediction study: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.700823/full
- Upper-limb torque prediction dataset: https://data.niaid.nih.gov/resources?id=zenodo_11209323
