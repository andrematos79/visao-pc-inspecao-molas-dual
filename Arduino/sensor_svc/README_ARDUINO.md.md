# Arduino Sensor Trigger – SVC

This module handles the presence detection for the SVC system.

## Hardware
- Arduino Uno
- IR Sensor E18-D80NK

## Wiring
- Signal → Pin 2
- VCC → 5V
- GND → GND

## Serial Communication
- Baud rate: 115200
- Output:
  - 1 → object detected
  - 0 → no object

## Function
Implements a digital presence detection used to trigger image capture and inference in the SVC system.