import csv
import time
import serial
from pathlib import Path
from datetime import datetime


# =========================================================
# 配置区
# =========================================================

SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 9600
TIMEOUT = 0.05

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "Logs"
LOG_FILE = LOG_DIR / "serial_handshake_log.csv"

FRAME_HEAD = 0xAA
FRAME_TAIL = 0x55

# 52RC -> 树莓派
MCU_TRIGGER_READY = 0xA1
MCU_ACK_RECEIVED = 0xCC
MCU_DONE = 0xDD
MCU_ERROR = 0xEE

# 树莓派 -> 52RC
CLASS_CODE = {
    "1": ("可回收", 0x01),
    "2": ("厨余", 0x02),
    "3": ("有害", 0x03),
    "4": ("其他", 0x04),
}


# =========================================================
# 工具函数
# =========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hex_str(data: bytes):
    return " ".join(f"{b:02X}" for b in data)


def write_log(event, detail="", rx="", tx="", result=""):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time", "event", "detail", "rx", "tx", "result"]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "time": now_str(),
            "event": event,
            "detail": detail,
            "rx": rx,
            "tx": tx,
            "result": result,
        })


def open_serial():
    print(f"正在打开串口：{SERIAL_PORT}, baud={BAUDRATE}")

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    time.sleep(1.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("串口打开成功。")
    write_log("open_serial", f"{SERIAL_PORT}, baud={BAUDRATE}", result="success")

    return ser


def read_one_byte(ser):
    data = ser.read(1)
    if data:
        return data[0]
    return None


def send_class_frame(ser, class_name, class_code):
    packet = bytes([FRAME_HEAD, class_code, FRAME_TAIL])
    ser.write(packet)
    ser.flush()

    print(f"已发送分类：{class_name} -> {hex_str(packet)}")
    write_log(
        event="send_class",
        detail=f"{class_name}, code=0x{class_code:02X}",
        tx=hex_str(packet),
        result="sent"
    )


def wait_for_byte(ser, target_byte, timeout_sec, name):
    start = time.time()

    while time.time() - start < timeout_sec:
        value = read_one_byte(ser)

        if value is None:
            continue

        print(f"收到字节：0x{value:02X}")
        write_log(
            event="rx_byte",
            detail=f"wait_for={name}",
            rx=f"0x{value:02X}"
        )

        if value == target_byte:
            return True

        if value == MCU_ERROR:
            print("收到 0xEE，52RC 报告错误。")
            return False

    print(f"等待 {name} 超时。")
    return False


def choose_class_from_terminal():
    print("\n请选择要发送给 52RC 的分类：")
    print("1. 可回收")
    print("2. 厨余")
    print("3. 有害")
    print("4. 其他")

    while True:
        choice = input("请输入 1/2/3/4，或 q 退出：").strip().lower()

        if choice == "q":
            return None, None

        if choice in CLASS_CODE:
            return CLASS_CODE[choice]

        print("输入无效，请重新输入。")


# =========================================================
# 主流程
# =========================================================

def main():
    print("========== 树莓派串口握手测试 ==========")
    print("流程：等待 0xA1 -> 手动选择分类 -> 发送 AA xx 55 -> 等待 0xCC -> 等待 0xDD")
    print("按 Ctrl+C 可退出。")

    ser = open_serial()

    try:
        while True:
            value = read_one_byte(ser)

            if value is None:
                continue

            print(f"\n收到字节：0x{value:02X}")
            write_log("rx_byte", rx=f"0x{value:02X}")

            if value == MCU_TRIGGER_READY:
                print("收到 0xA1：52RC 请求识别。")
                write_log("trigger", "received 0xA1", rx="0xA1")

                class_name, class_code = choose_class_from_terminal()

                if class_name is None:
                    print("用户退出。")
                    break

                send_class_frame(ser, class_name, class_code)

                ack_ok = wait_for_byte(
                    ser,
                    MCU_ACK_RECEIVED,
                    timeout_sec=2.0,
                    name="0xCC ACK"
                )

                if ack_ok:
                    print("收到 0xCC：52RC 已确认收到分类结果。")
                    write_log("ack", "received 0xCC", result="success")
                else:
                    print("未收到 0xCC。")
                    write_log("ack", "timeout or error", result="failed")
                    continue

                done_ok = wait_for_byte(
                    ser,
                    MCU_DONE,
                    timeout_sec=8.0,
                    name="0xDD DONE"
                )

                if done_ok:
                    print("收到 0xDD：52RC 动作完成。")
                    write_log("done", "received 0xDD", result="success")
                else:
                    print("未收到 0xDD。")
                    write_log("done", "timeout or error", result="failed")

                print("本轮握手结束，继续等待下一次 0xA1。")

            elif value == MCU_ERROR:
                print("收到 0xEE：52RC 报告错误。")
                write_log("mcu_error", "received 0xEE", rx="0xEE")

            else:
                print(f"未知字节：0x{value:02X}")
                write_log("unknown_byte", rx=f"0x{value:02X}")

    except KeyboardInterrupt:
        print("\n用户按 Ctrl+C 退出。")

    finally:
        ser.close()
        print("串口已关闭。")
        write_log("close_serial", result="closed")


if __name__ == "__main__":
    main()