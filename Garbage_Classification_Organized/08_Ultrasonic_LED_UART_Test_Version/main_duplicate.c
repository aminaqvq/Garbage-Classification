#include <REGX52.H>
#include <INTRINS.H>

typedef unsigned char u8;
typedef unsigned int u16;
typedef unsigned long u32;

/* Pin define */
sbit TRIG = P1^0;
sbit ECHO = P1^1;

sbit LED_RECOVERABLE = P2^0;
sbit LED_KITCHEN     = P2^1;
sbit LED_HARMFUL     = P2^2;
sbit LED_OTHER       = P2^3;
sbit LED_DETECT      = P2^4;

/* Protocol */
#define FRAME_HEAD 0xAA
#define FRAME_TAIL 0x55

#define MCU_TRIGGER_READY 0xA1
#define MCU_ACK_RECEIVED  0xCC
#define MCU_DONE          0xDD
#define MCU_ERROR         0xEE

#define CLASS_RECOVERABLE 0x01
#define CLASS_KITCHEN     0x02
#define CLASS_HARMFUL     0x03
#define CLASS_OTHER       0x04

/* Parameters */
#define DETECT_DISTANCE_CM 15
#define LED_ON_TIME_MS 3000
#define WAIT_PI_TIMEOUT_MS 10000
#define TRIGGER_INTERVAL_MS 2500
#define DETECT_LED_BLINK_MS 300

void DelayUs(u16 us)
{
    while (us--)
    {
        _nop_();
    }
}

void DelayMs(u16 ms)
{
    u16 i;
    u16 j;

    for (i = 0; i < ms; i++)
    {
        for (j = 0; j < 113; j++)
        {
            ;
        }
    }
}

void UartInit(void)
{
    TMOD &= 0x0F;
    TMOD |= 0x20;

    TH1 = 0xFD;
    TL1 = 0xFD;

    SCON = 0x50;
    PCON &= 0x7F;

    TR1 = 1;

    TI = 0;
    RI = 0;
}

void UartSendByte(u8 dat)
{
    SBUF = dat;

    while (TI == 0)
    {
        ;
    }

    TI = 0;
}

bit UartReadByteTimeout(u8 *dat, u16 timeout_ms)
{
    while (timeout_ms--)
    {
        if (RI)
        {
            RI = 0;
            *dat = SBUF;
            return 1;
        }

        DelayMs(1);
    }

    return 0;
}

void ClassLedAllOff(void)
{
    LED_RECOVERABLE = 1;
    LED_KITCHEN     = 1;
    LED_HARMFUL     = 1;
    LED_OTHER       = 1;
}

void DetectLedOn(void)
{
    LED_DETECT = 0;
}

void DetectLedOff(void)
{
    LED_DETECT = 1;
}

void DetectLedToggle(void)
{
    LED_DETECT = !LED_DETECT;
}

void LedShowClass(u8 garbage_type)
{
    ClassLedAllOff();

    switch (garbage_type)
    {
        case CLASS_RECOVERABLE:
            LED_RECOVERABLE = 0;
            break;

        case CLASS_KITCHEN:
            LED_KITCHEN = 0;
            break;

        case CLASS_HARMFUL:
            LED_HARMFUL = 0;
            break;

        case CLASS_OTHER:
            LED_OTHER = 0;
            break;

        default:
            break;
    }
}

void Timer0InitForMeasure(void)
{
    TMOD &= 0xF0;
    TMOD |= 0x01;

    TR0 = 0;
    TF0 = 0;
    TH0 = 0;
    TL0 = 0;
}

u16 Timer0GetCount(void)
{
    return ((u16)TH0 << 8) | TL0;
}

u16 UltrasonicMeasureCm(void)
{
    u16 count;
    u16 distance;
    u32 timeout_count;

    Timer0InitForMeasure();

    TRIG = 0;
    DelayUs(5);

    TRIG = 1;
    DelayUs(15);

    TRIG = 0;

    timeout_count = 0;

    while (ECHO == 0)
    {
        timeout_count++;

        if (timeout_count > 60000)
        {
            return 999;
        }
    }

    TH0 = 0;
    TL0 = 0;
    TF0 = 0;
    TR0 = 1;

    while (ECHO == 1)
    {
        if (TF0)
        {
            TR0 = 0;
            TF0 = 0;
            return 999;
        }
    }

    TR0 = 0;

    count = Timer0GetCount();

    distance = count / 54;

    return distance;
}

bit IsValidClassType(u8 garbage_type)
{
    if (
        garbage_type == CLASS_RECOVERABLE ||
        garbage_type == CLASS_KITCHEN ||
        garbage_type == CLASS_HARMFUL ||
        garbage_type == CLASS_OTHER
    )
    {
        return 1;
    }

    return 0;
}

bit ReceiveClassFrame(u8 *garbage_type)
{
    u8 b1;
    u8 b2;
    u8 b3;

    if (!UartReadByteTimeout(&b1, WAIT_PI_TIMEOUT_MS))
    {
        return 0;
    }

    if (b1 != FRAME_HEAD)
    {
        return 0;
    }

    if (!UartReadByteTimeout(&b2, 1000))
    {
        return 0;
    }

    if (!UartReadByteTimeout(&b3, 1000))
    {
        return 0;
    }

    if (b3 != FRAME_TAIL)
    {
        return 0;
    }

    if (!IsValidClassType(b2))
    {
        return 0;
    }

    *garbage_type = b2;

    return 1;
}

void SystemInit(void)
{
    UartInit();

    ClassLedAllOff();
    DetectLedOff();

    TRIG = 0;

    DelayMs(1000);
}

void main(void)
{
    u16 distance;
    u8 garbage_type;

    SystemInit();

    while (1)
    {
        distance = UltrasonicMeasureCm();

        if (distance <= DETECT_DISTANCE_CM)
        {
            DetectLedOn();

            UartSendByte(MCU_TRIGGER_READY);

            if (ReceiveClassFrame(&garbage_type))
            {
                UartSendByte(MCU_ACK_RECEIVED);

                LedShowClass(garbage_type);

                DelayMs(LED_ON_TIME_MS);

                ClassLedAllOff();

                UartSendByte(MCU_DONE);
            }
            else
            {
                UartSendByte(MCU_ERROR);
            }

            DetectLedOff();

            DelayMs(TRIGGER_INTERVAL_MS);
        }
        else
        {
            DetectLedToggle();
            DelayMs(DETECT_LED_BLINK_MS);
        }

        DelayMs(50);
    }
}