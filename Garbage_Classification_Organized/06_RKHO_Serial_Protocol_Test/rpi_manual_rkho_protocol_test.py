#!/usr/bin/env python3
"""手动 RKHO 串口协议测试。发送 R/K/H/O, 显示 T/F/N/D/E。"""
import argparse, serial, time, sys, select
from datetime import datetime

MCU={"T":"触发","F":"满载","N":"恢复","D":"完成","E":"错误"}
TX={"R":"可回收","K":"厨余","H":"有害","O":"其他"}

def main():
    p=argparse.ArgumentParser(description="RKHO 手动协议测试")
    p.add_argument("--serial-port",default="/dev/ttyAMA0")
    p.add_argument("--baudrate",type=int,default=9600)
    p.add_argument("--timeout",type=float,default=0.05)
    a=p.parse_args()
    ser=serial.Serial(a.serial_port,a.baudrate,timeout=a.timeout)
    time.sleep(1); ser.reset_input_buffer()
    print(f"串口: {a.serial_port} @ {a.baudrate}")
    print("MCU→RPi: T/F/N/D/E   RPi→MCU: R/K/H/O")
    print("输入 r/k/h/o 发送, q 退出\n")
    try:
        while True:
            d=ser.read(1)
            if d:
                ch=d.decode("ascii",errors="ignore").strip()
                if ch in MCU:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] RX: {ch} → {MCU[ch]}")
            r,_,_=select.select([sys.stdin],[],[],0)
            if r:
                cmd=sys.stdin.readline().strip().upper()
                if cmd=="Q": break
                if cmd in TX:
                    ser.write(cmd.encode("ascii")); ser.flush()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] TX: {cmd} → {TX[cmd]}")
                else: print(f"无效: {cmd}, 请用 R/K/H/O")
    except KeyboardInterrupt: pass
    finally: ser.close(); print("\n串口关闭")

if __name__=="__main__": main()
