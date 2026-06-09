#include <REGX52.H>

#define uchar unsigned char
#define uint unsigned int

sbit EN = P0^0;
sbit IN1 = P0^1;
sbit IN2 = P0^2;
sbit Servo = P1^5;
unsigned int cnt = 0;          
unsigned int angle_pwm = 10;   
int i;

void DelayMs(uint xms)
{
  uint i,j;
	
   for(i=xms; i>0; i--)
        for(j=110; j>0; j--);
}

uchar code step[8] = {
    0x10,  // 0001
    0x30,  // 0011
    0x20,  // 0010
    0x60,  // 0110
    0x40,  // 0100
    0xC0,  // 1100
    0x80,  // 1000
    0x90   // 1001
};

void Uart_Init()
{
	TMOD |= 0X20;
	TH1  = 0XFD;
	TL1 = 0XFD;
	TR1 = 1;
	SCON = 0X50;
	EA = 1;
	ES = 1;
}

void Timer0_Init(void) {
    TMOD &= 0xF0;
    TMOD |= 0x01;
    
    TH0 = 0xFF;
    TL0 = 0xD2;  
    
    ET0 = 1;
    EA = 1;
    TR0 = 1;
}

void Timer0_ISR(void) interrupt 1 {
    cnt++;
    
    TH0 = 0xFF;
    TL0 = 0xD2;   
    
    if(cnt <= angle_pwm)
    {    Servo = 1;
		}
		
    else{
        Servo = 0;}
   			
    if(cnt >= 400) {
        cnt = 0;
        Servo = 1;
    }
}

void Uart_IRQ() interrupt 4
{
	unsigned char ch;
	Timer0_Init();
	if(RI)
	{
		RI = 0;
		ch = SBUF;
		
		if(ch == 'R')
		{
			angle_pwm = 8;
			
		}
		if(ch == 'H')
		{
			angle_pwm = 19;
		
		}
		if(ch == 'K')
		{
			angle_pwm = 29;
		
		
		}
		if(ch == 'O')
		{
			angle_pwm = 36;
			
		}
	}
}
void main(void) {
//	  uint x;
    Timer0_Init();
    Uart_Init();
	  Servo = 1;
		EN = 0;
    while(1) {
       if(RI == 1){
				 RI = 0;
				 
				 EN = 1;
				 IN1 = 1;
				 IN2 = 0;
				 DelayMs(1100);
				 
				 EN = 0;
				 DelayMs(1000);
				 
				 EN = 1;
				 IN1 = 0;
				 IN2 = 1;
				 DelayMs(1100);
				 
				 EN = 0;
				
				RI = 0;
			}

    }
}

