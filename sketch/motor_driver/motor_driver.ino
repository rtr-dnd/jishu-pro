#include <SPI.h>
#include <MsTimer2.h>

// ピン定義。
#define PIN_SPI_MOSI 11
#define PIN_SPI_MISO 12
#define PIN_SPI_SCK 13
#define PIN_SPI_SS 10
#define PIN_BUSY 9

int input = -1;
int direction = 1;
int buf = 0;

void setup()
{
  delay(1000);
  pinMode(PIN_SPI_MOSI, OUTPUT);
  pinMode(PIN_SPI_MISO, INPUT);
  pinMode(PIN_SPI_SCK, OUTPUT);
  pinMode(PIN_SPI_SS, OUTPUT);
  pinMode(PIN_BUSY, INPUT);
  SPI.begin();
  SPI.setDataMode(SPI_MODE3);
  SPI.setBitOrder(MSBFIRST);
  Serial.begin(9600);
  digitalWrite(PIN_SPI_SS, HIGH);

  L6470_resetdevice(); //L6470リセット
  L6470_setup();       //L6470を設定

  MsTimer2::set(50, flash); //シリアルモニター用のタイマー割り込み
  MsTimer2::start();
  delay(4000);
}

void loop()
{
  input = Serial.read();
  if (input != -1)
  {
    switch (input)
    {
    case '-':
      direction = -1;
    case '0':
      buf = 0 + buf * 10;
      break;
    case '1':
      buf = 1 + buf * 10;
      break;
    case '2':
      buf = 2 + buf * 10;
      break;
    case '3':
      buf = 3 + buf * 10;
      break;
    case '4':
      buf = 4 + buf * 10;
      break;
    case '5':
      buf = 5 + buf * 10;
      break;
    case '6':
      buf = 6 + buf * 10;
      break;
    case '7':
      buf = 7 + buf * 10;
      break;
    case '8':
      buf = 8 + buf * 10;
      break;
    case '9':
      buf = 9 + buf * 10;
      break;
    case 'a':
      Serial.print("registered");
      Serial.print(direction);
      Serial.print(buf);
      L6470_move(direction, buf); //反転
      buf = 0;
      direction = 1;
    case 'r':
      buf = 0;
      direction = 1;
    default:
      break;
    }
  }
  // L6470_move(1, 5000);  //正転
  // L6470_busydelay(500); //2秒待つ
  // L6470_move(-1, 2000); //反転
  // L6470_busydelay(500); //2秒待つ
  // L6470_move(-1, 3000); //反転
  // L6470_busydelay(500); //2秒待つ
}

void L6470_setup()
{
  L6470_setparam_acc(0x8A);      //[R, WS] 加速度default 0x08A (12bit) (14.55*val+14.55[step/s^2])
  L6470_setparam_dec(0x8A);      //[R, WS] 減速度default 0x08A (12bit) (14.55*val+14.55[step/s^2])
  L6470_setparam_maxspeed(0x40); //[R, WR]最大速度default 0x041 (10bit) (15.25*val+15.25[step/s])
  L6470_setparam_minspeed(0x01); //[R, WS]最小速度default 0x000 (1+12bit) (0.238*val[step/s])
  L6470_setparam_fsspd(0x3ff);   //[R, WR]μステップからフルステップへの切替点速度default 0x027 (10bit) (15.25*val+7.63[step/s])
  L6470_setparam_kvalhold(0x20); //[R, WR]停止時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
  L6470_setparam_kvalrun(0x20);  //[R, WR]定速回転時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
  L6470_setparam_kvalacc(0x20);  //[R, WR]加速時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
  L6470_setparam_kvaldec(0x20);  //[R, WR]減速時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)

  L6470_setparam_stepmood(0x03); //ステップモードdefault 0x07 (1+3+1+3bit)
}

void flash()
{
  Serial.print("0x");
  Serial.print(L6470_getparam_abspos(), HEX);
  Serial.print("  ");
  Serial.print("0x");
  Serial.println(L6470_getparam_speed(), HEX);
}
