# Data Schema

## Intent Labels

- `outside_bathroom`: Dog may want to go outside to urinate or defecate.
- `food_water`: Dog may want food or water.
- `play`: Dog may want active play.
- `attention`: Dog may want social contact or engagement.
- `stress_discomfort`: Dog may be stressed, uncomfortable, or in distress.
- `unknown`: Intent is unclear or model confidence is low.

## Context Fields

`location_context` examples:

- `door`
- `kitchen`
- `couch`
- `crate`
- `outside`
- `unknown`

`situation_context` examples:

- `before_walk`
- `before_food`
- `owner_leaving`
- `stranger_nearby`
- `toy_visible`
- `unknown`

## Dog

- `id`: Integer primary key.
- `name`: Dog name.
- `breed`: Optional breed.
- `age`: Optional age in years.
- `notes`: Optional owner notes.
- `created_at`: Creation timestamp.

## AudioClip

- `id`: Integer primary key.
- `dog_id`: Dog foreign key.
- `file_path`: Stored audio path.
- `duration_seconds`: Audio duration if readable.
- `recorded_at`: Recording timestamp.
- `location_context`: Owner-selected location context.
- `situation_context`: Owner-selected situation context.
- `owner_label`: Owner's best label.
- `outcome_label`: Optional later outcome label.
- `prediction_label`: Latest predicted likely intent.
- `prediction_confidence`: Latest prediction confidence.
- `confirmed_correct`: Owner confirmation when available.
- `notes`: Optional notes.
- `created_at`: Creation timestamp.

## Prediction

- `id`: Integer primary key.
- `clip_id`: Audio clip foreign key.
- `predicted_label`: Predicted likely intent.
- `confidence`: Model confidence.
- `top_3_json`: Top likely labels and probabilities as JSON.
- `model_version`: Model version string.
- `created_at`: Creation timestamp.

