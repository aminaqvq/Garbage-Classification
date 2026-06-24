/*********************************************************************
 * STC89C52RC 最终单片机固件 — 五分类视觉触发版
 *
 * 功能：
 *   接收树莓派 R/K/H/O 分类命令
 *   控制舵机角度 + 电机推出/回位
 *   满载传感器检测 (L0-L3) + F/N 反馈
 *   分拣完成后发送 D
 *   非法字符发送 E
 *   满载时禁止动作并返回 F
 *
 * 协议（ASCII 单字符）：
 *   RPi → MCU:  R(可回收) K(厨余) H(有害) O(其他)
 *   MCU → RPi:  D(完成) F(满载) N(恢复) E(错误)
 *
 * 重要约束：
 *   1. P0 口开漏结构，作为普通 IO 需外部上拉电阻（4.7k~10kΩ）
 *   2. 舵机/电机必须独立供电（不要从 52RC 取电）
 *   3. 树莓派 GPIO 3.3V，52RC 5V，UART 需电平转换模块
 *   4. 不含 HC-SR04 超声波，不含 T 触发逻辑
 *   5. 兼容 Keil C51，所有局部变量在函数开头声明
 *
 * 部署目录：09_Vision_Trigger_5Class_System/mcu/
 *********************************************************************/

#include <REG52.H>

/* ── 类型定义 ────────────────────────────── */
typedef unsigned char  u8;
typedef unsigned int   u16;

/* ── 引脚定义 ────────────────────────────── */
/* 直流电机驱动 (L298N / L9110S) */
sbit MOTOR_EN  = P0^0;   /* 使能 */
sbit MOTOR_IN1 = P0^1;   /* 方向 1 */
sbit MOTOR_IN2 = P0^2;   /* 方向 2 */

/* 舵机 PWM (软件 PWM 使用 Timer0 中断) */
sbit SERVO = P1^5;

/* 满载传感器 — 低电平触发（遮挡/满载时传感器输出低电平）
 * ⚠ P0 口开漏，必须外部上拉 4.7k~10kΩ 到 VCC */
sbit FULL_L0 = P0^3;
sbit FULL_L1 = P0^5;
sbit FULL_L2 = P0^6;
sbit FULL_L3 = P0^7;

/* ── 协议字符定义 ────────────────────────── */
#define MCU_DONE_CHAR    'D'   /* 分拣完成 */
#define MCU_FULL_CHAR    'F'   /* 满载暂停 */
#define MCU_NORMAL_CHAR  'N'   /* 满载解除 */
#define MCU_ERROR_CHAR   'E'   /* 错误 */

/* ── 舵机角度定义 (PWM 脉宽值) ──────────── */
/* 注意：角度值需根据实际舵机微调！ */
#define SERVO_RECOVERABLE_PWM   8    /* 可回收 */
#define SERVO_HARMFUL_PWM      19    /* 有害   */
#define SERVO_KITCHEN_PWM      29    /* 厨余   */
#define SERVO_OTHER_PWM        36    /* 其他   */
#define SERVO_HOME_PWM         10    /* 回中位 */

/* ── 系统参数 ────────────────────────────── */
#define FULL_DEBOUNCE_MS       50    /* 满载消抖毫秒 */
#define DOOR_OPEN_MS         1000    /* 开门时间 */
#define DOOR_STOP_MS         1000    /* 停止时间 */
#define DOOR_CLOSE_MS        1000    /* 关门时间 */
#define SERVO_HOLD_MS         500    /* 舵机保持时间 */

/* ── 全局变量 ────────────────────────────── */
volatile u8   uart_rx_char = 0;
volatile bit  uart_rx_flag = 0;

unsigned int  servo_cnt = 0;
unsigned int  servo_pwm = 10;     /* 当前舵机 PWM 值 */

bit  full_flag      = 0;          /* 当前满载状态 */
bit  last_full_raw  = 0;          /* 上次满载原始值（消抖用） */

/* ── 函数声明 ────────────────────────────── */
void DelayMs(u16 ms);
void Uart_Init(void);
void Uart_SendByte(u8 dat);
bit  IsFullRaw(void);
void CheckFullLoad(void);
void MotorStop(void);
void MotorOpen(void);
void MotorClose(void);
u8   ServoPwmByClassChar(u8 ch);
void ServoSetAngle(u8 pwm_val);
void ServoHoldMs(u8 pwm_val, u16 ms);
void SortAction(u8 ch);
void ProcessUartCommand(void);

/* ── 延时 ────────────────────────────────── */
void DelayMs(u16 ms)
{
    u16 i, j;
    for (i = ms; i > 0; i--)
        for (j = 110; j > 0; j--);
}

/* ── 串口初始化 (Timer1, 9600bps@11.0592MHz) */
void Uart_Init(void)
{
    TMOD |= 0x20;         /* T1 模式2，8位自动重装 */
    TH1   = 0xFD;         /* 9600 bps */
    TL1   = 0xFD;
    TR1   = 1;
    SCON  = 0x50;         /* 模式1，允许接收 */
    EA    = 1;
    ES    = 1;
    TI    = 0;
    RI    = 0;
}

/* ── 定时器0初始化（舵机 PWM，50Hz）─────── */
void Timer0_Init(void)
{
    TMOD &= 0xF0;
    TMOD |= 0x01;         /* T0 模式1，16位 */
    TH0   = 0xFF;
    TL0   = 0xD2;         /* ≈50us 周期 */
    ET0   = 1;
    EA    = 1;
    TR0   = 1;
}

/* ── 定时器0中断：舵机 PWM ───────────────── */
void Timer0_ISR(void) interrupt 1
{
    TH0 = 0xFF;
    TL0 = 0xD2;
    servo_cnt++;

    if (servo_cnt <= servo_pwm)
        SERVO = 1;
    else
        SERVO = 0;

    if (servo_cnt >= 400) {
        servo_cnt = 0;
        SERVO = 1;
    }
}

/* ── 串口发送 ────────────────────────────── */
void Uart_SendByte(u8 dat)
{
    bit saved_es;
    saved_es = ES;
    ES = 0;
    SBUF = dat;
    while (TI == 0);
    TI = 0;
    ES = saved_es;
}

/* ── 串口中断：只接收 ────────────────────── */
void Uart_IRQ(void) interrupt 4
{
    if (RI) {
        RI = 0;
        uart_rx_char = SBUF;
        uart_rx_flag = 1;
    }
}

/* ── 满载传感器原始读取 ──────────────────── */
/* 低电平触发：任意一个传感器为 0 → 满载 */
bit IsFullRaw(void)
{
    if (FULL_L0 == 0 || FULL_L1 == 0 || FULL_L2 == 0 || FULL_L3 == 0)
        return 1;
    else
        return 0;
}

/* ── 满载检测与 F/N 发送 ─────────────────── */
void CheckFullLoad(void)
{
    bit raw;
    raw = IsFullRaw();

    if (raw != last_full_raw) {
        DelayMs(FULL_DEBOUNCE_MS);      /* 消抖 */
        raw = IsFullRaw();
    }

    if (raw != last_full_raw) {
        last_full_raw = raw;

        if (raw == 1 && full_flag == 0) {
            /* 从未满载 → 满载 */
            full_flag = 1;
            MotorStop();
            Uart_SendByte(MCU_FULL_CHAR);
        }
        else if (raw == 0 && full_flag == 1) {
            /* 从满载 → 恢复正常 */
            full_flag = 0;
            Uart_SendByte(MCU_NORMAL_CHAR);
        }
    }
}

/* ── 电机控制 ────────────────────────────── */
void MotorStop(void)
{
    MOTOR_EN  = 0;
    MOTOR_IN1 = 0;
    MOTOR_IN2 = 0;
}

void MotorOpen(void)
{
    MOTOR_EN  = 1;
    MOTOR_IN1 = 1;
    MOTOR_IN2 = 0;
    DelayMs(DOOR_OPEN_MS);
    MotorStop();
    DelayMs(DOOR_STOP_MS);
}

void MotorClose(void)
{
    MOTOR_EN  = 1;
    MOTOR_IN1 = 0;
    MOTOR_IN2 = 1;
    DelayMs(DOOR_CLOSE_MS);
    MotorStop();
}

/* ── 舵机角度映射 ────────────────────────── */
u8 ServoPwmByClassChar(u8 ch)
{
    if (ch == 'R')      return SERVO_RECOVERABLE_PWM;
    else if (ch == 'H') return SERVO_HARMFUL_PWM;
    else if (ch == 'K') return SERVO_KITCHEN_PWM;
    else if (ch == 'O') return SERVO_OTHER_PWM;
    return SERVO_HOME_PWM;
}

/* ── 舵机设置角度 ────────────────────────── */
void ServoSetAngle(u8 pwm_val)
{
    servo_pwm = pwm_val;
}

/* ── 舵机保持指定时间 ────────────────────── */
void ServoHoldMs(u8 pwm_val, u16 ms)
{
    servo_pwm = pwm_val;
    DelayMs(ms);
}

/* ── 分拣动作 ────────────────────────────── */
void SortAction(u8 ch)
{
    u8 pwm;
    pwm = ServoPwmByClassChar(ch);

    /* 1. 舵机转向对应垃圾桶 */
    ServoHoldMs(pwm, SERVO_HOLD_MS);
    DelayMs(300);

    /* 2. 电机推出 */
    MotorOpen();

    /* 3. 舵机回中位 */
    ServoHoldMs(SERVO_HOME_PWM, SERVO_HOLD_MS);
    DelayMs(200);

    /* 4. 电机关门 */
    MotorClose();
}

/* ── 处理串口命令 ────────────────────────── */
void ProcessUartCommand(void)
{
    u8 ch;

    if (uart_rx_flag == 0)
        return;

    uart_rx_flag = 0;
    ch = uart_rx_char;

    /* 忽略 MCU 自己发送的 F/N/D/E 回显 */
    if (ch == MCU_FULL_CHAR || ch == MCU_NORMAL_CHAR ||
        ch == MCU_DONE_CHAR || ch == MCU_ERROR_CHAR) {
        return;
    }

    /* 合法分类命令：R / K / H / O */
    if (ch == 'R' || ch == 'H' || ch == 'K' || ch == 'O') {
        if (full_flag == 1) {
            /* 满载：禁止动作，返回 F */
            MotorStop();
            Uart_SendByte(MCU_FULL_CHAR);
            return;
        }

        /* 未满载：执行分拣动作 */
        SortAction(ch);

        /* 动作完成后发送 D */
        Uart_SendByte(MCU_DONE_CHAR);
        return;
    }

    /* 非法字符：发送 E */
    Uart_SendByte(MCU_ERROR_CHAR);
}

/* ── 主函数 ──────────────────────────────── */
void main(void)
{
    /* 初始化 */
    Timer0_Init();
    Uart_Init();

    /* 硬件初始状态 */
    SERVO     = 1;
    MOTOR_EN  = 0;
    full_flag     = 0;
    last_full_raw = IsFullRaw();

    while (1) {
        CheckFullLoad();         /* 检测满载/恢复，必要时发送 F/N */
        ProcessUartCommand();    /* 处理树莓派发来的 R/H/K/O */
    }
}
