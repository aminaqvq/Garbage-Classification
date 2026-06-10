# 09 — 锁定触发版本（T/A/D/E 协议）

## 1. 这个文件夹是干什么的

树莓派上位机的另一种实现方式：**等待 MCU 发送触发字符 'T' 后才进行一次 AI 识别**，识别完成后发送分类字符，然后等待 MCU 回复 A/D/E 确认。与最终版 AA 帧协议不同，此版本使用 ASCII 单字符协议。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `locked_trigger_system.py` | RPi Python | T/A/D/E 锁定触发上位机 | ⚠ 无匹配固件 |

## 3. 工作流程

```
1. 树莓派等待 MCU 发送字符 'T'（超声波触发信号）
2. 收到 T → 锁定当前帧 → 开始 AI 识别
3. AI 识别稳定 → 发送 R/K/H/O 分类字符
4. 暂停识别，等待 MCU 回复：
   - 'A' = ACK 已确认
   - 'D' = DONE 动作完成
   - 'E' = ERROR 异常
5. 收到 D 或超时 → 回到等待 T 状态
```

## 4. 串口协议

| 方向 | 字符 | 含义 |
|------|------|------|
| MCU → RPi | T | Trigger：超声波检测到物体 |
| RPi → MCU | R/K/H/O | 分类结果 |
| MCU → RPi | A | ACK：已收到分类 |
| MCU → RPi | D | DONE：动作完成 |
| MCU → RPi | E | ERROR：异常 |

## 5. 运行方法

```bash
cd 09_Locked_Trigger_Version/

# 触发模式（等待下位机发 T）
python3 locked_trigger_system.py --mode trigger

# 手动模式（按 r/k/h/o 手动测试）
python3 locked_trigger_system.py --mode manual
```

## 6. 关键配置参数

| 参数 | 说明 |
|------|------|
| `--mode trigger/manual` | 触发/手动模式 |
| `--serial-port` | 串口设备（默认 /dev/ttyAMA0） |
| `--conf-threshold` | 置信度阈值（默认 0.80） |
| `--min-predict-sec` | 收到 T 后最短识别时间 |
| `--save-on-send` | 发送时自动截图 |
| `--no-window` | SSH 模式运行 |

## 7. ⚠ 注意事项 — 无匹配固件

**这是最高风险项，请务必阅读：**

- **`locked_trigger_system.py` 需要 MCU 主动发送 'T'（触发）、'A'（ACK）、'D'（DONE）、'E'（ERROR）**
- **当前 `firmware/` 下所有 C 文件中没有任何一个实现了 T/A/D/E 协议**
- 最接近的固件是 `mcu_full/main.c`（支持 R/H/K/O + F/N），但它不发 T/A/D/E
- 可能此脚本原本配合了未提交到仓库的固件版本
- **如果你要使用此版本，需要自己修改 MCU 固件以发送 T/A/D/E 字符**
- 不建议作为部署版本使用
