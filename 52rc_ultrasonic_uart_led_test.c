#include <REGX52.H>
#include <INTRINS.H>


typedef unsigned char u8;
typedef unsigned int u16;
typedef unsigned long u32;


// =========================================================
// 1. 引脚定义
// =========================================================

// HC-SR04 超声波模块
sbit TRIG = P1^0;
sbit ECHO = P1^1;


// P2.0 ~ P2.3：分类 LED，低电平点亮
sbit LED_RECOVERABLE = P2^0;  // 可回收
sbit LED_KITCHEN     = P2^1;  // 厨余
sbit LED_HARMFUL     = P2^2;  // 有害
sbit LED_OTHER       = P2^3;  // 其他

// P2.4：物体检测状态 LED，低电平点亮
// 未检测到物体：闪烁
// 检测到物体：常亮
sbit LED_DETECT      = P2^4;


// =========================================================
// 2. 串口协议定义
// =========================================================

#define FRAME_HEAD 0xAA
#define FRAME_TAIL 0x55

// 52RC -> 树莓派
#define MCU_TRIGGER_READY 0xA1
#define MCU_ACK_RECEIVED  0xCC
#define MCU_DONE          0xDD
#define MCU_ERROR         0xEE

// 树莓派 -> 52RC
#define CLASS_RECOVERABLE 0x01    // 可回收
#define CLASS_KITCHEN     0x02    // 厨余
#define CLASS_HARMFUL     0x03    // 有害
#define CLASS_OTHER       0x04    // 其他


// =========================================================
// 3. 参数配置
// =========================================================

// 超声波检测距离，小于等于该距离认为有物体，单位 cm
#define DETECT_DISTANCE_CM 15

// 每次触发后间隔一段时间，防止连续重复触发
#define TRIGGER_INTERVAL_MS 2500

// 分类 LED 点亮时间
#define LED_ON_TIME_MS 1500

// 等待树莓派返回分类结果的最长时间
// 第一阶段手动测试时建议长一点，方便你在终端输入 1/2/3/4
// 后面正式 AI 自动识别时可以改成 5000
#define WAIT_PI_TIMEOUT_MS 30000

// 没检测到物体时，P2.4 状态灯闪烁间隔
#define DETECT_LED_BLINK_MS 300


// =========================================================
// 4. 延时函数
// =========================================================

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
        for (j = 0; j < 113; j++);
    }
}


// =========================================================
// 5. UART 串口初始化
// 11.0592MHz，9600bps
// Timer1 mode2
// =========================================================

void UartInit(void)
{
    // Timer1 设置为 8 位自动重装模式
    // 保留 Timer0 的配置位
    TMOD &= 0x0F;
    TMOD |= 0x20;

    // 9600bps @ 11.0592MHz
    TH1 = 0xFD;
    TL1 = 0xFD;

    // 串口模式 1，允许接收
    SCON = 0x50;

    // SMOD = 0
    PCON &= 0x7F;

    TR1 = 1;

    TI = 0;
    RI = 0;
}


void UartSendByte(u8 dat)
{
    SBUF = dat;

    while (TI == 0);

    TI = 0;
}


void UartSendBytes(u8 *buf, u8 len)
{
    u8 i;

    for (i = 0; i < len; i++)
    {
        UartSendByte(buf[i]);
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


// =========================================================
// 6. LED 控制
// 低电平点亮，高电平熄灭
// =========================================================

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


// =========================================================
// 7. Timer0 用于超声波 Echo 计时
// Timer0 mode1，16 位计数
// 11.0592MHz 下，机器周期约 1.085us
// 距离 cm 约等于 count / 54
// =========================================================

void Timer0InitForMeasure(void)
{
    // Timer0 mode1，16 位定时器
    // 保留 Timer1 的配置
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


// =========================================================
// 8. HC-SR04 超声波测距
// 返回距离，单位 cm
// 如果超时，返回 999
// =========================================================

u16 UltrasonicMeasureCm(void)
{
    u16 count;
    u16 distance;
    u32 timeout_count;

    Timer0InitForMeasure();

    // 发送 10us 以上触发脉冲
    TRIG = 0;
    DelayUs(5);

    TRIG = 1;
    DelayUs(15);

    TRIG = 0;

    // 等待 Echo 变高，防止死等
    timeout_count = 0;
    while (ECHO == 0)
    {
        timeout_count++;

        if (timeout_count > 60000)
        {
            return 999;
        }
    }

    // Echo 变高，开始计时
    TH0 = 0;
    TL0 = 0;
    TF0 = 0;
    TR0 = 1;

    // 等待 Echo 变低，或者 Timer0 溢出
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

    // 11.0592MHz 下：
    // 一个计数约 1.085us
    // HC-SR04 距离 cm ≈ 高电平时间 us / 58
    // 所以距离约为 count * 1.085 / 58 ≈ count / 54
    distance = count / 54;

    return distance;
}


// =========================================================
// 9. 接收树莓派分类帧
// 格式：AA xx 55
// 成功返回 1，失败返回 0
// =========================================================

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

    // 等待帧头
    if (!UartReadByteTimeout(&b1, WAIT_PI_TIMEOUT_MS))
    {
        return 0;
    }

    if (b1 != FRAME_HEAD)
    {
        return 0;
    }

    // 等待分类码
    if (!UartReadByteTimeout(&b2, 1000))
    {
        return 0;
    }

    // 等待帧尾
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


// =========================================================
// 10. 系统初始化
// =========================================================

void SystemInit(void)
{
    UartInit();

    ClassLedAllOff();
    DetectLedOff();

    TRIG = 0;

    DelayMs(1000);
}


// =========================================================
// 11. 主程序
// =========================================================

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
            // 检测到物体：P2.4 常亮
            DetectLedOn();

            // 通知树莓派：检测到物体，请开始识别
            UartSendByte(MCU_TRIGGER_READY);

            // 等待树莓派返回分类帧 AA xx 55
            if (ReceiveClassFrame(&garbage_type))
            {
                // 已正确收到树莓派分类结果
                UartSendByte(MCU_ACK_RECEIVED);

                // 点亮对应分类 LED
                LedShowClass(garbage_type);

                DelayMs(LED_ON_TIME_MS);

                // 关闭分类 LED
                ClassLedAllOff();

                // 本轮动作完成
                UartSendByte(MCU_DONE);
            }
            else
            {
                // 接收失败、超时、帧格式错误
                UartSendByte(MCU_ERROR);
            }

            // 本轮结束，检测状态灯熄灭
            DetectLedOff();

            // 防止连续重复触发
            DelayMs(TRIGGER_INTERVAL_MS);
        }
        else
        {
            // 未检测到物体：P2.4 闪烁
            DetectLedToggle();
            DelayMs(DETECT_LED_BLINK_MS);
        }

        DelayMs(50);
    }
}