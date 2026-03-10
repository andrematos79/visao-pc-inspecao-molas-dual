/*
==========================================================
SVC - Sensor de Presença para Disparo de Inferência
Sistema de Visão Computacional para Inspeção de Molas

Autor: André Gama de Matos
Projeto: SVC - Sistema de Visão Computacional
Ano: 2026
==========================================================

Função:
Detectar presença de peça via sensor E18-D80NK e enviar
estado para o computador via comunicação serial.

Saída serial esperada:
PRESENT=1  -> peça detectada
PRESENT=0  -> peça ausente

Baudrate homologado: 115200
Placa utilizada: Arduino Uno
==========================================================
*/

const int SENSOR_PIN = 2;   // pino digital conectado ao sensor
int sensorState = 0;
int lastState = -1;

void setup()
{
  pinMode(SENSOR_PIN, INPUT);

  // Inicializa comunicação serial
  Serial.begin(115200);

  // Pequeno delay para estabilização
  delay(1000);
}

void loop()
{
  sensorState = digitalRead(SENSOR_PIN);

  // Envia apenas quando o estado muda
  if (sensorState != lastState)
  {
    if (sensorState == HIGH)
    {
      Serial.println("PRESENT=1");
    }
    else
    {
      Serial.println("PRESENT=0");
    }

    lastState = sensorState;
  }

  delay(20); // pequena estabilização de leitura
}