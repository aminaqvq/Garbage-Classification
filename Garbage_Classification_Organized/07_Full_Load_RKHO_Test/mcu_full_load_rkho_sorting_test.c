#include <REGX52.H>
#define uchar unsigned char
#define uint unsigned int

/* ── 引脚定义 ────────────────────────────── */
/* 直流电机 */
sbit EN  = P0^0;   // 使能
sbit IN1 = P0^1;   // 方向1
sbit IN2 = P0^2;   // 方向2

/* 舵机 PWM */
sbit Servo = P1^5;

/* 满载传感器 — 低电平触发 */
sbit L0 = P0^3;
sbit L1 = P0^5;
sbit L2 = P0^6;
sbit L3 = P0^7;

/* ── 全局变量 ────────────────────────────── */
unsigned int cnt = 0;
unsigned int angle_pwm = 10;   // 初始角度

bit full_flag      = 0;        // 当前是否满载
bit last_full_raw  = 0;        // 上一次原始满载状态，用于检测变化

volatile unsigned char uart_rx_char = 0;
volatile bit uart_rx_flag = 0;

/* ── 延时函数 ────────────────────────────── */
void DelayMs(uint xms)
{
    uint i, j;
    for(i = xms; i > 0; i--)
        for(j = 110; j > 0; j--);
}

/* ── 串口初始化 ──────────────────────────── */
void Uart_Init(void)
{
    TMOD |= 0x20;     // T1 模式2（8位自动重装）
    TH1   = 0xFD;     // 9600 bps @ 11.0592 MHz
    TL1   = 0xFD;
    TR1   = 1;
    SCON  = 0x50;     // 模式1，允许接收
    EA    = 1;
    ES    = 1;
}

/* ── 定时器0初始化（舵机 PWM）───────────── */
void Timer0_Init(void)
{
    TMOD &= 0xF0;
    TMOD |= 0x01;       // T0 模式1（16位）
    TH0   = 0xFF;
    TL0   = 0xD2;       // ≈ 50 us 周期
    ET0   = 1;
    EA    = 1;
    TR0   = 1;
}

/* ── 定时器0中断：舵机 PWM ───────────────── */
void Timer0_ISR(void) interrupt 1
{
    cnt++;
    TH0 = 0xFF;
    TL0 = 0xD2;

    if(cnt <= angle_pwm)
        Servo = 1;
    else
        Servo = 0;

    if(cnt >= 400) {
        cnt = 0;
        Servo = 1;
    }
}

/* ── 串口发送一个字节 ────────────────────── */
/*
 * 临时关闭 ES → 写入 SBUF → 等待 TI → 清除 TI → 恢复 ES。
 * 避免串口中断和 TI 竞争。
 */
void Uart_SendByte(unsigned char dat)
{
    bit saved_es = ES;
    ES = 0;
    SBUF = dat;
    while(TI == 0);
    TI = 0;
    ES = saved_es;
}

/* ── 串口中断：只负责接收 ───────────────── */
/*
 * 不在中断里做耗时操作！
 * 只把收到的字节存到 uart_rx_char 并置标志位 uart_rx_flag。
 */
void Uart_IRQ(void) interrupt 4
{
    if(RI) {
        RI = 0;
        uart_rx_char = SBUF;
        uart_rx_flag = 1;
    }
}

/* ── 满载传感器原始读取 ──────────────────── */
/*
 * 低电平触发：任意一个传感器为 0 → 认为满载。
 * 返回 1 = 满载， 0 = 未满载。
 */
unsigned char IsFullRaw(void)
{
    if(L0 == 0 || L1 == 0 || L2 == 0 || L3 == 0)
        return 1;
    else
        return 0;
}

/* ── 满载检测与 F / N 发送 ──────────────── */
/*
 * 带消抖：状态变化后延时 50 ms 再确认一次。
 *
 * F = FULL   → 满载，树莓派应暂停分类
 * N = NORMAL → 恢复，垃圾桶已清理
 */
void CheckFullLoad(void)
{
    unsigned char raw = IsFullRaw();

    if(raw != last_full_raw) {
        DelayMs(50);                // 消抖
        raw = IsFullRaw();
    }

    if(raw != last_full_raw) {
        last_full_raw = raw;

        if(raw == 1 && full_flag == 0) {
            /* 从未满载 → 满载 */
            full_flag = 1;
            EN = 0;                 // 停止电机
            Uart_SendByte('F');     // 通知上位机：满载
        }
        else if(raw == 0 && full_flag == 1) {
            /* 从满载 → 恢复正常 */
            full_flag = 0;
            Uart_SendByte('N');     // 通知上位机：恢复正常
        }
    }
}

/* ── 开门动作（门控流程）─────────────────── */
/*
 * 如果满载，立即返回，不开门。
 * 流程：开门 → 停 → 关门 → 停。
 * 结束后确保 EN = 0。
 */
void DoorAction(void)
{
    if(full_flag == 1)
        return;                     // 满载时禁止开门

    /* 开门 */
    EN  = 1;
    IN1 = 1;
    IN2 = 0;
    DelayMs(1000);

    /* 停止 */
    EN = 0;
    DelayMs(1000);

    /* 关门 */
    EN  = 1;
    IN1 = 0;
    IN2 = 1;
    DelayMs(1000);

    /* 停止 */
    EN = 0;
}

/* ── 处理来自上位机的串口命令 ───────────── */
/*
 * 主循环调用。
 *
 * 满载时：忽略 R/H/K/O，不改变舵机、不开门，
 *          重新发送 F 提醒上位机。
 * 未满载：按字符设置 angle_pwm 并执行开门。
 *
 * 字符映射：
 *   R → angle_pwm = 8   (可回收)
 *   H → angle_pwm = 19  (有害)
 *   K → angle_pwm = 29  (厨余)
 *   O → angle_pwm = 36  (其他)
 */
void ProcessUartCommand(void)
{
    unsigned char ch;

    if(uart_rx_flag == 0)
        return;

    uart_rx_flag = 0;
    ch = uart_rx_char;

    if(ch == 'F' || ch == 'N') {
        /* 下位机自己发送 F/N，不对收到的 F/N 做响应 */
        return;
    }

    if(ch == 'R' || ch == 'H' || ch == 'K' || ch == 'O') {
        if(full_flag == 1) {
            /* 满载：忽略分类命令，停止电机，再发一次 F 提醒 */
            EN = 0;
            Uart_SendByte('F');
            return;
        }

        /* 未满载：正常设置舵机角度并执行开门 */
        if(ch == 'R')      angle_pwm = 8;
        else if(ch == 'H') angle_pwm = 19;
        else if(ch == 'K') angle_pwm = 29;
        else if(ch == 'O') angle_pwm = 36;

        DoorAction();
    }
}

/* ── 主函数 ──────────────────────────────── */
void main(void)
{
    Timer0_Init();
    Uart_Init();
    Servo = 1;
    EN    = 0;
    full_flag     = 0;
    last_full_raw = IsFullRaw();

    while(1) {
        CheckFullLoad();       // 检测满载/恢复，必要时发送 F/N
        ProcessUartCommand();  // 处理上位机发来的 R/H/K/O
    }
}
