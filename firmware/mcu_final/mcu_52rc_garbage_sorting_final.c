#include <REGX52.H>
#include <INTRINS.H>

/*
 * 52RC / STC89C52RC 单片机最终版：超声波触发 + UART 通信 + LED 指示 + 舵机分拣 + 电机推出/回位
 *
 * 晶振假设：11.0592 MHz
 * 串口：9600 bps, 8N1
 *
 * 完整流程：
 * 1) HC-SR04 检测到物体 <= DETECT_DISTANCE_CM。
 * 2) 单片机发送 0xA1 给树莓派，请求 AI 识别。
 * 3) 单片机等待树莓派返回 AA xx 55。
 * 4) 收到合法分类后回复 0xCC。
 * 5) 舵机转到对应垃圾桶方向，电机推出垃圾并回位。
 * 6) 动作完成后回复 0xDD。
 */

typedef unsigned char u8;
typedef unsigned int  u16;
typedef unsigned long u32;

/* =========================================================
 * 1. 引脚定义
 * ========================================================= */

/* HC-SR04 超声波模块 */
sbit TRIG = P1^0;
sbit ECHO = P1^1;

/* 舵机信号线 */
sbit SERVO = P1^5;

/* 直流电机驱动：兼容 L298N / TB6612 / 继电器式正反转模块
 * EN=使能，IN1/IN2=方向。
 * 注意：经典 8051 的 P0 口是开漏结构，必要时请加外部上拉，或改到 P1/P2 口。
 */
sbit MOTOR_EN  = P0^0;
sbit MOTOR_IN1 = P0^1;
sbit MOTOR_IN2 = P0^2;

/* P2.0 ~ P2.3：分类 LED，低电平点亮 */
sbit LED_RECOVERABLE = P2^0;  /* 可回收 */
sbit LED_KITCHEN     = P2^1;  /* 厨余 */
sbit LED_HARMFUL     = P2^2;  /* 有害 */
sbit LED_OTHER       = P2^3;  /* 其他 */

/* P2.4：检测状态 LED，低电平点亮；未检测到物体时闪烁，检测到物体时常亮 */
sbit LED_DETECT      = P2^4;


/* =========================================================
 * 2. 串口协议定义
 * ========================================================= */

#define FRAME_HEAD 0xAA
#define FRAME_TAIL 0x55

/* 52RC -> 树莓派 */
#define MCU_TRIGGER_READY 0xA1
#define MCU_ACK_RECEIVED  0xCC
#define MCU_DONE          0xDD
#define MCU_ERROR         0xEE

/* 树莓派 -> 52RC */
#define CLASS_RECOVERABLE 0x01
#define CLASS_KITCHEN     0x02
#define CLASS_HARMFUL     0x03
#define CLASS_OTHER       0x04


/* =========================================================
 * 3. 参数配置：根据实物微调这里
 * ========================================================= */

/* 超声波检测距离，小于等于该距离认为有物体，单位 cm */
#define DETECT_DISTANCE_CM      15

/* 检测到一次后，等待一段时间，防止同一个垃圾重复触发 */
#define TRIGGER_INTERVAL_MS     2500

/* 等待树莓派返回分类帧的最长时间。AI 自动识别一般 4 秒内，留 8 秒更稳。 */
#define WAIT_PI_TIMEOUT_MS      8000

/* 没检测到物体时，状态灯闪烁间隔 */
#define DETECT_LED_BLINK_MS     300

/* 舵机 PWM 周期约 20ms。这里沿用你原舵机代码的 angle_pwm 单位：1 单位约 50us。
 * 例如 19 表示高电平约 950us，36 表示高电平约 1800us。
 * 如果实物方向不对，只需要改下面 4 个分类角度。
 */
#define SERVO_HOME_PWM          10
#define SERVO_RECOVERABLE_PWM   8
#define SERVO_HARMFUL_PWM       19
#define SERVO_KITCHEN_PWM       29
#define SERVO_OTHER_PWM         36

/* 舵机转到分类位置后等待时间 */
#define SERVO_SETTLE_MS         700

/* 分拣电机动作时间：正转推出、停顿、反转回位 */
#define MOTOR_PUSH_MS           1100
#define MOTOR_PAUSE_MS          350
#define MOTOR_RETURN_MS         1100


/* =========================================================
 * 4. 延时函数
 * ========================================================= */

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


/* =========================================================
 * 5. UART 串口：Timer1 mode2，9600bps @ 11.0592MHz
 * ========================================================= */

void UartInit(void)
{
    /* Timer1 使用 8 位自动重装模式；保留 Timer0 的配置位 */
    TMOD &= 0x0F;
    TMOD |= 0x20;

    TH1 = 0xFD;
    TL1 = 0xFD;

    /* 串口模式 1，允许接收 */
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

void UartClearRx(void)
{
    u8 dummy;

    if (RI)
    {
        RI = 0;
        dummy = SBUF;
        dummy = dummy;
    }
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


/* =========================================================
 * 6. LED 控制：低电平点亮
 * ========================================================= */

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


/* =========================================================
 * 7. 超声波测距：Timer0 mode1，16 位计数
 * 11.0592MHz 下机器周期约 1.085us，距离 cm 约 count / 54
 * ========================================================= */

void Timer0InitForMeasure(void)
{
    /* Timer0 mode1，保留 Timer1 配置 */
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

bit ObjectDetectedStable(void)
{
    u8 i;
    u8 hit_count;
    u16 distance;

    hit_count = 0;

    for (i = 0; i < 3; i++)
    {
        distance = UltrasonicMeasureCm();
        if (distance <= DETECT_DISTANCE_CM)
        {
            hit_count++;
        }
        DelayMs(60);
    }

    if (hit_count >= 2)
    {
        return 1;
    }

    return 0;
}


/* =========================================================
 * 8. 接收树莓派分类帧：AA xx 55
 * ========================================================= */

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
    u16 remain_ms;
    bit head_found;

    remain_ms = WAIT_PI_TIMEOUT_MS;
    head_found = 0;

    /* 等待帧头。这里会忽略偶发噪声，直到看到 0xAA 或超时。 */
    while (remain_ms--)
    {
        if (UartReadByteTimeout(&b1, 1))
        {
            if (b1 == FRAME_HEAD)
            {
                head_found = 1;
                break;
            }
        }
    }

    if (!head_found)
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


/* =========================================================
 * 9. 舵机与电机动作
 * ========================================================= */

u8 ServoPwmByClass(u8 garbage_type)
{
    switch (garbage_type)
    {
        case CLASS_RECOVERABLE:
            return SERVO_RECOVERABLE_PWM;

        case CLASS_KITCHEN:
            return SERVO_KITCHEN_PWM;

        case CLASS_HARMFUL:
            return SERVO_HARMFUL_PWM;

        case CLASS_OTHER:
            return SERVO_OTHER_PWM;

        default:
            return SERVO_HOME_PWM;
    }
}

void ServoPulseOnce(u8 pwm_units)
{
    u16 high_us;
    u16 low_us;

    high_us = ((u16)pwm_units) * 50;

    if (high_us < 300)
    {
        high_us = 300;
    }

    if (high_us > 2500)
    {
        high_us = 2500;
    }

    low_us = 20000 - high_us;

    SERVO = 1;
    DelayUs(high_us);
    SERVO = 0;

    DelayMs(low_us / 1000);
    DelayUs(low_us % 1000);
}

void ServoHoldMs(u8 pwm_units, u16 hold_ms)
{
    u16 cycles;

    cycles = hold_ms / 20;
    if (cycles == 0)
    {
        cycles = 1;
    }

    while (cycles--)
    {
        ServoPulseOnce(pwm_units);
    }
}

void MotorStop(void)
{
    MOTOR_EN  = 0;
    MOTOR_IN1 = 0;
    MOTOR_IN2 = 0;
}

void MotorForwardStart(void)
{
    MOTOR_IN1 = 1;
    MOTOR_IN2 = 0;
    MOTOR_EN  = 1;
}

void MotorBackwardStart(void)
{
    MOTOR_IN1 = 0;
    MOTOR_IN2 = 1;
    MOTOR_EN  = 1;
}

void SortAction(u8 garbage_type)
{
    u8 target_pwm;

    target_pwm = ServoPwmByClass(garbage_type);

    /* 先让舵机转到对应垃圾桶口 */
    ServoHoldMs(target_pwm, SERVO_SETTLE_MS);

    /* 推出垃圾。电机运行期间继续给舵机脉冲，防止舵机松动。 */
    MotorForwardStart();
    ServoHoldMs(target_pwm, MOTOR_PUSH_MS);

    MotorStop();
    ServoHoldMs(target_pwm, MOTOR_PAUSE_MS);

    /* 机构回位 */
    MotorBackwardStart();
    ServoHoldMs(target_pwm, MOTOR_RETURN_MS);

    MotorStop();

    /* 舵机回默认位置 */
    ServoHoldMs(SERVO_HOME_PWM, 600);
    SERVO = 0;
}


/* =========================================================
 * 10. 系统初始化
 * ========================================================= */

void SystemInit(void)
{
    UartInit();

    ClassLedAllOff();
    DetectLedOff();

    TRIG = 0;
    SERVO = 0;
    MotorStop();

    /* 上电后让舵机先回默认位置 */
    ServoHoldMs(SERVO_HOME_PWM, 800);

    DelayMs(500);
}


/* =========================================================
 * 11. 主程序
 * ========================================================= */

void main(void)
{
    u16 distance;
    u8 garbage_type;

    SystemInit();

    while (1)
    {
        distance = UltrasonicMeasureCm();

        if (distance <= DETECT_DISTANCE_CM && ObjectDetectedStable())
        {
            DetectLedOn();
            UartClearRx();

            /* 通知树莓派开始识别 */
            UartSendByte(MCU_TRIGGER_READY);

            /* 等待树莓派返回 AA xx 55 */
            if (ReceiveClassFrame(&garbage_type))
            {
                UartSendByte(MCU_ACK_RECEIVED);

                LedShowClass(garbage_type);
                SortAction(garbage_type);
                ClassLedAllOff();

                UartSendByte(MCU_DONE);
            }
            else
            {
                MotorStop();
                ClassLedAllOff();
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
