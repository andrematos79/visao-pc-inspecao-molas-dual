# Arduino do SVC

## Função
Receber o sinal do sensor de presença e transmitir o estado via serial para o SVC.

## Placa
Arduino Uno

## Baudrate homologado
115200

## Saída serial esperada
PRESENT=0
PRESENT=1

## Observações
- O código permanece gravado na placa mesmo ao trocar de PC.
- Portanto, ao trocar o PC da estação de inspeção, não é necessário regravar o firmware do Arduino, apenas conectar o dispositivo e identificar a porta COM correta.
- O Arduino IDE não deve permanecer com o Serial Monitor aberto durante o uso do SVC.
- Confirmar a porta COM correta no Gerenciador de Dispositivos.

## Ligação do Sensor (E18-D80NK)
Sensor	Arduino
Marrom	5V
Azul	GND
Preto	Pino Digital 2

## Observação importante:

O E18-D80NK normalmente é saída NPN (LOW quando detecta).
Se o seu estiver invertido, basta trocar no código: if (sensorState == LOW)

## Saída esperada no Serial Monitor:

PRESENT=0
PRESENT=1
PRESENT=0
PRESENT=1

