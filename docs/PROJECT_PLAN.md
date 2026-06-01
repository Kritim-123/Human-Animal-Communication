# DogBridge Project Plan

## Phase 1: Data Collection MVP

- Build the backend for dog profiles, clip uploads, context labels, predictions, confirmations, and stats.
- Build the mobile app skeleton for collecting real dog audio.
- Store owner labels and outcome confirmations.
- Keep language honest: DogBridge estimates likely intent; it does not translate dog language.

## Phase 2: Dog-Specific Baseline

- Train a baseline model on confirmed clips.
- Evaluate per dog and per intent label.
- Use owner corrections to improve the dataset.
- Report uncertainty and return `unknown` when confidence is low.

## Phase 3: Multimodal Context

- Add collar IMU movement signals.
- Add phone camera posture analysis.
- Add richer location/context metadata.
- Add time-of-day and routine patterns.

## Phase 4: Bidirectional Communication

- Let the phone send tone or vibration cues.
- Help dogs learn cue meanings through repetition.
- Track response outcomes.
- Keep cues simple and ethically safe.

## Phase 5: Collar Prototype

- Prototype with ESP32-S3 or Raspberry Pi Zero 2 W.
- Include MEMS microphone, IMU, vibration motor, small speaker or buzzer, BLE/Wi-Fi, and rechargeable battery.
- Design around waterproofing, comfort, battery life, and privacy.

