/*********************************************************************
 * 52RC / STC89C52RC 最终单片机固件
 *
 * 功能整合：
 *    HC-SR04 超声波检测 + 满载传感器 L0-L3 + 舵机分拣
 *    + 电机推出/回位 + LED 指示 + ASCII RKHO 协议
 *
 * 最终协议（ASCII 单字符，双向，不再使用 AA xx 55）：
 *
 *   MCU → RPi:
 *     T = Trigger (超声波检测到物体，请求 AI 识别)
 *     F = Full    (垃圾桶满载，暂停分类)
 *     N = Normal  (满载解除，恢复正常)
 *     D = Done    (动作完成)
 *     E = Error   (超时/异常/非法命令)
 *
 *   RPi → MCU:
 *     R = 可回收 (Recyclable)
 *     K = 厨余   (Kitchen)
 *     H = 有害   (Harmful)
 *     O = 其他   (Other)
 *
 * 注意：
 *   1. P0 口是开漏结构，P0.0~P0.7 作为普通 IO 可能需要外部上拉电阻。
 *   2. 舵机/电机必须独立供电，不要从 52RC 取电。
 *   3. 电平转换：树莓派 GPIO 3.3V，52RC 5V，串口需电平转换。
 *********************************************************************/
#include <REGX52.H>
#include <INTRINS.H>

typedef unsigned char  u8;
typedef unsigned int   u16;

/* ── 引脚定义 ────────────────────────────────── */
/* HC-SR04 超声波 */
sbit TRIG = P1^0;
sbit ECHO = P1^1;

/* 舵机 — 软件 PWM（不使用 Timer0 中断） */
sbit SERVO = P1^5;

/* 直流电机驱动 (L298N / L9110S)
 * EN=使能, IN1/IN2=方向 */
sbit MOTOR_EN  = P0^0;
sbit MOTOR_IN1 = P0^1;
sbit MOTOR_IN2 = P0^2;

/* 满载传感器 — 低电平触发
 * ⚠ P0 口开漏, 接传感器时需要外部上拉 */
sbit L0 = P0^3;
sbit L1 = P0^5;
sbit L2 = P0^6;
sbit L3 = P0^7;

/* 分类 LED — 低电平点亮 */
sbit LED_RECOVERABLE = P2^0;  /* 可回收 */
sbit LED_KITCHEN     = P2^1;  /* 厨余 */
sbit LED_HARMFUL     = P2^2;  /* 有害 */
sbit LED_OTHER       = P2^3;  /* 其他 */

/* 检测状态 LED — 未检测到物体时闪烁, 检测到时常亮 */
sbit LED_DETECT      = P2^4;

/* ── 协议字符定义（最终 ASCII RKHO 协议）────── */
#define MCU_TRIGGER_CHAR    'T'
#define MCU_FULL_CHAR       'F'
#define MCU_NORMAL_CHAR     'N'
#define MCU_DONE_CHAR       'D'
#define MCU_ERROR_CHAR      'E'

#define CLASS_R_CHAR 'R'   /* 可回收 */
#define CLASS_K_CHAR 'K'   /* 厨余 */
#define CLASS_H_CHAR 'H'   /* 有害 */
#define CLASS_O_CHAR 'O'   /* 其他 */

/* ── 舵机角度定义 ───────────────────────────── */
#define SERVO_RECOVERABLE_PWM   8
#define SERVO_HARMFUL_PWM      19
#define SERVO_KITCHEN_PWM      29
#define SERVO_OTHER_PWM        36
#define SERVO_HOME_PWM         10

/* ── 系统参数 ────────────────────────────────── */
#define DETECT_DISTANCE_CM     15    /* 超声波检测距离阈值（厘米）*/
#define DETECT_STABLE_COUNT     3    /* 连续命中次数才确认检测到 */
#define FULL_DEBOUNCE_MS       50    /* 满载状态变化消抖毫秒 */
#define TRIGGER_COOLDOWN_MS  3000    /* 发 T 后冷却时间, 避免同一物体重复触发 */
#define WAIT_CLASS_TIMEOUT_MS 5000   /* 等 R/K/H/O 超时 */
#define DOOR_OPEN_MS         1000    /* 开门时间 */
#define DOOR_STOP_MS         1000    /* 停止时间 */
#define DOOR_CLOSE_MS        1000    /* 关门时间 */
#define DETECT_BLINK_MS       300    /* 检测 LED 闪烁周期 */

/* ── 全局变量 ────────────────────────────────── */
volatile u8  uart_rx_char = 0;
volatile bit uart_rx_flag = 0;

bit full_flag      = 0;       /* 当前是否满载 */
bit last_full_raw  = 0;       /* 上一次满载原始状态 */
u16 detect_blink_timer = 0;

/* ── 函数声明 ────────────────────────────────── */
void DelayMs(u16 ms);
void Uart_Init(void);
void Uart_SendByte(u8 dat);
u16  UltrasonicMeasureCm(void);
bit  ObjectDetectedStable(void);
bit  IsFullRaw(void);
void CheckFullLoad(void);
void MotorStop(void);
void MotorPush(void);
void MotorReturn(void);
void AllClassLedOff(void);
void LedShowClass(u8 ch);
void ServoPulseOnce(u16 pwm_val);
void ServoHoldMs(u16 pwm_val, u16 ms);
u8   ServoPwmByClassChar(u8 ch);
void SortActionByClassChar(u8 ch);
bit  ReceiveClassChar(u8 *ch);
void ProcessSystem(void);

/* ── 延时 ────────────────────────────────────── */
void DelayMs(u16 ms) {
    u16 i, j;
    for (i = ms; i > 0; i--)
        for (j = 110; j > 0; j--);
}

/* ── 串口初始化 (Timer1, 9600bps) ───────────── */
void Uart_Init(void) {
    TMOD |= 0x20;      /* T1 模式2 8位自动重装 */
    TH1  = 0xFD;
    TL1  = 0xFD;
    TR1  = 1;
    SCON = 0x50;       /* 模式1, 允许接收 */
    EA   = 1;
    ES   = 1;
    TI   = 0;
    RI   = 0;
}

/* ── 串口发送（临时关 ES 防冲突）───────────── */
void Uart_SendByte(u8 dat) {
    bit saved_es = ES;
    ES = 0;
    SBUF = dat;
    while (!TI);
    TI = 0;
    ES = saved_es;
}

/* ── 串口中断：只负责接收 ──────────────────── */
void Uart_IRQ(void) interrupt 4 {
    if (RI) {
        RI = 0;
        uart_rx_char = SBUF;
        uart_rx_flag = 1;
    }
}

/* ── 超声波测距 (Timer0 用于 Echo 计时) ─────── */
u16 UltrasonicMeasureCm(void) {
    u16 time_val;

    /* 发送 10us 触发脉冲 */
    TRIG = 1;
    _nop_(); _nop_(); _nop_(); _nop_(); _nop_();
    _nop_(); _nop_(); _nop_(); _nop_(); _nop_();
    TRIG = 0;

    /* 等待 Echo 变高 (超时保护) */
    u16 timeout = 5000;
    while (!ECHO && timeout) timeout--;
    if (!timeout) return 999;

    /* Timer0 清零并启动 */
    TMOD &= 0xF0;
    TMOD |= 0x01;        /* T0 模式1 (16位) */
    TH0 = 0;
    TL0 = 0;
    TR0 = 1;

    /* 等待 Echo 变低 (超时保护) */
    timeout = 5000;
    while (ECHO && timeout) timeout--;

    TR0 = 0;
    time_val = (TH0 << 8) | TL0;

    if (!timeout) return 999;

    /* 距离(cm) = 时间(us) / 58 */
    return (u16)(time_val * 1.085f / 58.0f);
}

/* ── 稳定检测物体 ───────────────────────────── */
bit ObjectDetectedStable(void) {
    u8 hit_count = 0;
    u8 i;
    for (i = 0; i < DETECT_STABLE_COUNT; i++) {
        if (UltrasonicMeasureCm() <= DETECT_DISTANCE_CM)
            hit_count++;
        DelayMs(30);
    }
    return (hit_count >= DETECT_STABLE_COUNT);
}

/* ── 满载传感器原始读取（低电平触发）───────── */
bit IsFullRaw(void) {
    if (L0 == 0 || L1 == 0 || L2 == 0 || L3 == 0)
        return 1;
    else
        return 0;
}

/* ── 满载检测与 F/N 发送 ───────────────────── */
void CheckFullLoad(void) {
    bit raw = IsFullRaw();

    if (raw != last_full_raw) {
        DelayMs(FULL_DEBOUNCE_MS);        /* 消抖 */
        raw = IsFullRaw();
    }

    if (raw != last_full_raw) {
        last_full_raw = raw;

        if (raw == 1 && full_flag == 0) {
            /* 满载触发 */
            full_flag = 1;
            MotorStop();
            Uart_SendByte(MCU_FULL_CHAR);
        }
        else if (raw == 0 && full_flag == 1) {
            /* 满载解除 */
            full_flag = 0;
            Uart_SendByte(MCU_NORMAL_CHAR);
        }
    }
}

/* ── 电机控制 ────────────────────────────────── */
void MotorStop(void) {
    MOTOR_EN  = 0;
    MOTOR_IN1 = 0;
    MOTOR_IN2 = 0;
}

void MotorPush(void) {
    MOTOR_EN  = 1;
    MOTOR_IN1 = 1;
    MOTOR_IN2 = 0;
    DelayMs(DOOR_OPEN_MS);
    MotorStop();
    DelayMs(DOOR_STOP_MS);
}

void MotorReturn(void) {
    MOTOR_EN  = 1;
    MOTOR_IN1 = 0;
    MOTOR_IN2 = 1;
    DelayMs(DOOR_CLOSE_MS);
    MotorStop();
}

/* ── LED 控制 ────────────────────────────────── */
void AllClassLedOff(void) {
    LED_RECOVERABLE = 1;
    LED_KITCHEN     = 1;
    LED_HARMFUL     = 1;
    LED_OTHER       = 1;
}

void LedShowClass(u8 ch) {
    AllClassLedOff();
    if (ch == CLASS_R_CHAR)      LED_RECOVERABLE = 0;
    else if (ch == CLASS_K_CHAR) LED_KITCHEN     = 0;
    else if (ch == CLASS_H_CHAR) LED_HARMFUL     = 0;
    else if (ch == CLASS_O_CHAR) LED_OTHER       = 0;
}

/* ── 软件舵机脉冲（不依赖中断）─────────────── */
void ServoPulseOnce(u16 pwm_val) {
    SERVO = 1;
    /* 高电平时间 = pwm_val * (255 - 0xD2 + 1) / 12MHz * 12 ≈ pwm_val * 46us */
    u8 i;
    for (i = 0; i < pwm_val; i++) {
        _nop_(); _nop_(); _nop_(); _nop_();
        _nop_(); _nop_(); _nop_(); _nop_();
    }
    SERVO = 0;
}

void ServoHoldMs(u16 pwm_val, u16 ms) {
    u16 tick;
    u16 pulses = ms * 20;  /* 约 20 次/秒 (50Hz) */
    for (tick = 0; tick < pulses; tick++) {
        ServoPulseOnce(pwm_val);
        DelayMs(20);
    }
}

/* ── 舵机角度映射 ────────────────────────────── */
u8 ServoPwmByClassChar(u8 ch) {
    if (ch == CLASS_R_CHAR)      return SERVO_RECOVERABLE_PWM;
    else if (ch == CLASS_H_CHAR) return SERVO_HARMFUL_PWM;
    else if (ch == CLASS_K_CHAR) return SERVO_KITCHEN_PWM;
    else if (ch == CLASS_O_CHAR) return SERVO_OTHER_PWM;
    return SERVO_HOME_PWM;
}

/* ── 分拣动作 ────────────────────────────────── */
void SortActionByClassChar(u8 ch) {
    u8 pwm = ServoPwmByClassChar(ch);

    /* 舵机转向对应垃圾桶 */
    ServoHoldMs(pwm, 500);
    DelayMs(500);

    /* 电机推出 */
    MotorPush();

    /* 舵机回位 */
    ServoHoldMs(SERVO_HOME_PWM, 500);

    /* 电机回位 */
    MotorReturn();
}

/* ── 接收分类字符（带超时和满载检测）───────── */
bit ReceiveClassChar(u8 
