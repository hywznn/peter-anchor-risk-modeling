# Modeling Notes

## Why Real Public Data Replaced Synthetic Rows

Safety-related modeling should not present generated sensor rows as if they were measured data. The current project therefore uses public real sensor recordings where possible and explicitly separates unavailable field data from model inputs.

The real-data pipeline uses wearable IMU recordings for fall and daily-activity motion, plus historical wind data for environmental context. Peter Anchor-specific rope tension, harness pressure, and suction readiness are not synthesized because they require actual hardware or field collection.

## Why Windowed IMU Features Fit This Stage

The selected IMU dataset provides time-series acceleration, angular velocity, and inclination signals. Instead of classifying raw rows independently, the pipeline aggregates samples into sliding windows and extracts motion features such as peak acceleration, acceleration variance, peak angular velocity, body-angle proxy, and sway proxy.

This better matches the actual detection problem: a fall-risk signal is not one isolated sample, but a short motion pattern.

## Why Group-aware Evaluation Matters

Randomly splitting windows can leak nearly identical adjacent windows into both training and test sets. The real-data trainer therefore splits by recording file, so a recording used for testing is not also represented in training.

This is still not field validation, but it is more honest than a row-level random split.

## Why Recall Still Matters

In a safety context, missing fall-like motion is worse than over-alerting on some normal activity. Accuracy alone can hide this failure mode, so the real-data metrics include fall recall and macro F1.

## Data Still Required for Productization

- Peter Anchor rope tension time-series data
- Harness pressure sensor data from real wear tests
- Suction readiness and emergency anchor deployment logs
- Worker motion data collected during actual rope-access or exterior-cleaning workflows
- Building-face wind measurements near the worker, not only regional weather data
- Field pilot labels reviewed by safety experts
- Safety certification and mechanical load-test results

## Interpretation Boundary

The real-data model only demonstrates that public IMU recordings can support a fall-risk classifier. It does not prove Peter Anchor's safety performance, does not validate the hardware concept, and does not replace field testing.
