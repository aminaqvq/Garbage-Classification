/* mcu_rkho_protocol_test.c — 最小 RKHO 协议测试固件 */
#include <REGX52.H>
typedef unsigned char u8; typedef unsigned int u16;
void DelayMs(u16 m){u16 i,j;for(i=m;i>0;i--)for(j=110;j>0;j--);}
void UartI(){TMOD|=0x20;TH1=0xFD;TL1=0xFD;TR1=1;SCON=0x50;EA=1;ES=1;}
void UartS(u8 d){ES=0;SBUF=d;while(!TI);TI=0;ES=1;}
volatile u8 rx=0; volatile bit rf=0;
void UartIRQ(void) interrupt 4 {if(RI){RI=0;rx=SBUF;rf=1;}}
void main(){UartI(); u16 t=0; u8 ch; while(1){t++;
if(t>=500){t=0;UartS('T');}
if(rf){rf=0;ch=rx;if(ch=='R'||ch=='K'||ch=='H'||ch=='O'){DelayMs(200);UartS('D');}else UartS('E');}
DelayMs(1);}}
