# Future Hardware Plan

## Collar Architecture

- Microphone: MEMS microphone for short audio clips and eventual real-time inference.
- IMU: Accelerometer and gyroscope for posture, movement, scratching, pacing, and activity patterns.
- Haptic motor: Gentle vibration cues for bidirectional communication experiments.
- Buzzer or speaker: Simple tones for learned cues.
- Battery: Rechargeable battery sized for comfort and safe runtime.
- Phone connection: BLE for local communication, with Wi-Fi as an option for higher bandwidth prototypes.
- Enclosure: Lightweight, waterproof, chew-resistant, and comfortable.

## Prototype Boards

- ESP32-S3 for low-power BLE/Wi-Fi and embedded audio experiments.
- Raspberry Pi Zero 2 W for heavier local prototyping if battery life is less important.

## Privacy Considerations

- Audio may capture humans and household sounds.
- Store only clips the owner explicitly records.
- Make deletion easy.
- Prefer local processing where feasible.
- Clearly label when microphones are active.

## Safety Considerations

- Keep the collar light.
- Avoid heat buildup.
- Use low-intensity haptics and tones.
- Do not use cues as punishment.
- Validate enclosure materials for skin safety and water resistance.

