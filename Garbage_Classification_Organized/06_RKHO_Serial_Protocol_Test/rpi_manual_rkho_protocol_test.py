#!/usr/bin/env python3
"""手动 RKHO 串口协议测试。发送 R/K/H/O，显示 D/F/N/E。
注意：新版视觉触发系统不使用 T 触发字符，T 在此仅作 [legacy] 显示。
"""
import argparse
import sys
import time
from datetime import datetime

# MCU→RPi 字符映射；T 为 [legacy]，新版不用
MCU_RX_MAP = {"T": "触发 [legacy]", "F": "满载", "N": "恢复", "D": "完成", "E": "错误"}
# RPi→MCU 字符映射
TX_MAP = {"R": "可回收", "K": "厨余", "H": "有害", "O": "其他"}


def main():
    p = argparse.ArgumentParser(description="RKHO 手动协议测试")
    p.add_argument("--serial-port", default="/dev/ttyAMA0", help="串口设备")
    p.add_argument("--baudrate", type=int, default=9600, help="波特率")
    p.add_argument("--timeout", type=float, default=0.05, help="串口超时")
    args = p.parse_args()

    # 延迟导入 pyserial — 只有真正需要串口时才加载
    try:
        import serial as _serial_mod
    except ImportError:
        print("错误：未安装 pyserial。请运行：pip install pyserial")
        sys.exit(1)

    ser = _serial_mod.Serial(args.serial_port, args.baudrate, timeout=args.timeout)
    time.sleep(1)
    ser.reset_input_buffer()
    print(f"串口: {args.serial_port} @ {args.baudrate}")
    print("MCU→RPi: D/F/N/E (+ T [legacy])")
    print("RPi→MCU: R/K/H/O")
    print("输入 r/k/h/o 发送, q 退出\n")
    try:
        while True:
            d = ser.read(1)
            if d:
                ch = d.decode("ascii", errors="ignore").strip()
                if ch in MCU_RX_MAP:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] RX: {ch} → {MCU_RX_MAP[ch]}")
            r = _select_stdin()
            if r:
                cmd = r.strip().upper()
                if cmd == "Q":
                    break
                if cmd in TX_MAP:
                    ser.write(cmd.encode("ascii"))
                    ser.flush()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] TX: {cmd} → {TX_MAP[cmd]}")
                else:
                    print(f"无效: {cmd}, 请用 R/K/H/O")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\n串口关闭")


def _select_stdin():
    """跨平台非阻塞读取 stdin 一行。"""
    import select
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        return sys.stdin.readline()
    return None


if __name__ == "__main__":
    main()
