import serial
import serial.tools.list_ports
import time

print("Portas disponíveis:")

ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p.device)

PORT = input("\nDigite a porta COM (ex: COM4): ")

print("\nAbrindo porta...")

ser = serial.Serial(PORT, 115200, timeout=1)

time.sleep(2)

print("Lendo dados da serial...\n")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print("RECEBIDO:", line)