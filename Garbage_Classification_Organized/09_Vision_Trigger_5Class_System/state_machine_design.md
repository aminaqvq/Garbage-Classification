# 状态机设计文档 — 五分类视觉触发版

> 版本：vision_trigger_5class_v1
> 状态：最终可运行系统
>
> 状态机已在 `rpi/rpi_vision_trigger_sorting.py` 中实现。

---

## 1. 状态列表

| 状态名 | 说明 |
|--------|------|
| **BOOT** | 系统启动初始化，加载模型、打开串口/摄像头 |
| **IDLE_WAIT_VISUAL** | 空闲等待：模型持续推理，检测画面中是否有垃圾出现 |
| **CANDIDATE_DETECTED** | 候选垃圾检测：连续 N 帧稳定识别到四类垃圾之一 |
| **SEND_SORT_COMMAND** | 发送分拣命令：向 MCU 发送 R/K/H/O 单个字符 |
| **WAIT_MCU_DONE** | 等待 MCU 完成：等待 MCU 返回 D（分拣完成），带超时保护 |
| **WAIT_RETURN_TO_PENDING** | 等待回到待分拣：等待画面稳定回到「待分拣」状态后才允许下一轮 |
| **FULL_PAUSED** | 满载暂停：MCU 返回 F，树莓派暂停所有分拣命令 |
| **ERROR_RECOVERY** | 错误恢复：MCU 返回 E 或超时，进入错误恢复逻辑 |

---

## 2. 状态转移图

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
  ┌──────┐    ┌─────────────┐    ┌──────────────────┐  │
  │ BOOT │───▶│ IDLE_WAIT   │───▶│ CANDIDATE        │  │
  └──────┘    │ _VISUAL     │    │ _DETECTED        │  │
              └──────┬──────┘    └────────┬─────────┘  │
                     │                    │ 稳定≥5帧    │
                     │         ┌──────────┘             │
                     │         ▼                        │
                     │  ┌──────────────────┐            │
                     │  │ SEND_SORT_       │            │
                     │  │ COMMAND          │            │
                     │  └────────┬─────────┘            │
                     │           │ 发送 R/K/H/O         │
                     │           ▼                      │
                     │  ┌──────────────────┐            │
                     │  │ WAIT_MCU_        │◀───────────┤ E 或超时
                     │  │ DONE             │────────────┤──────────┐
                     │  └────────┬─────────┘   F 满载     │          │
                     │           │ D 完成                  │          ▼
                     │           ▼                         │  ┌──────────────┐
                     │  ┌──────────────────┐              │  │ ERROR_       │
                     └──│ WAIT_RETURN_     │              │  │ RECOVERY     │
                        │ TO_PENDING       │              │  └──────────────┘
                        └────────┬─────────┘              │
                                 │ 稳定回到「待分拣」     │
                                 └────────────────────────┘

  ┌──────────────┐
  │ FULL_PAUSED  │◀──── MCU 返回 F（可从多个状态进入）
  └──────┬───────┘
         │ MCU 返回 N
         ▼
  ┌──────────────┐
  │ IDLE_WAIT    │  （满载解除后回到空闲等待）
  │ _VISUAL      │
  └──────────────┘
```

### 满载中断路径

任何状态收到 MCU 返回 `F` → 立即转入 **FULL_PAUSED**。  
**FULL_PAUSED** 收到 MCU 返回 `N` → 转入 **IDLE_WAIT_VISUAL**，恢复运行。

---

## 3. 关键规则

### 规则 1：「待分拣」不发送命令
「待分拣」是视觉等待状态，表示画面中没有需要分拣的垃圾。模型输出「待分拣」时，树莓派**不发送任何分拣命令**给 MCU。  
`action_mapping` 中不包含「待分拣」，`no_action_classes` 明确列出。

### 规则 2：只有稳定识别到四类垃圾才发送命令
仅当模型连续稳定识别到「其他」「厨余」「可回收」「有害」之一（默认 ≥5 帧）时，树莓派才进入 **SEND_SORT_COMMAND**。

### 规则 3：每个垃圾只允许触发一次分拣
通过 `send_cooldown_seconds` 冷却时间和 `WAIT_RETURN_TO_PENDING` 状态确保同一垃圾不会被重复分拣。

### 规则 4：发送后必须等待 D 或进入超时保护
发送 R/K/H/O 后进入 **WAIT_MCU_DONE**：
- 收到 `D` → 进入 **WAIT_RETURN_TO_PENDING**
- 超时（默认 8 秒）→ 进入 **ERROR_RECOVERY**

### 规则 5：发送后必须等待画面回到「待分拣」才允许下一次分拣
**WAIT_RETURN_TO_PENDING** 状态下持续推理，只有当模型连续稳定输出「待分拣」（默认 ≥8 帧）后才转回 **IDLE_WAIT_VISUAL**。  
此规则与 `require_return_to_pending_before_next_sort = true` 对应。

### 规则 6：MCU 返回 F 时，树莓派必须暂停发送
任何状态收到 `F` → 转入 **FULL_PAUSED**，禁止发送任何 R/K/H/O。

### 规则 7：MCU 返回 N 时，树莓派才恢复发送
**FULL_PAUSED** 状态下收到 `N` → 转入 **IDLE_WAIT_VISUAL**，恢复正常运行。

### 规则 8：新版不需要 T 字符触发
新版系统由树莓派摄像头持续推理主动判断，不再等待 MCU 发 `T` 触发识别。  
旧版 `WAIT_TRIGGER` 状态已废弃。

### 规则 9：新版不需要超声波
MCU 不再搭载 HC-SR04 超声波传感器，不再运行 `UltrasonicMeasureCm()` 和 `ObjectDetectedStable()` 函数。  
视觉触发完全替代距离传感器触发。

---

## 4. 状态转移条件速查

| 当前状态 | 转移条件 | 下一状态 |
|----------|----------|----------|
| BOOT | 初始化完成 | IDLE_WAIT_VISUAL |
| IDLE_WAIT_VISUAL | 连续 ≥5 帧识别到四类垃圾之一 | CANDIDATE_DETECTED |
| CANDIDATE_DETECTED | 条件满足 | SEND_SORT_COMMAND |
| SEND_SORT_COMMAND | 发送完成 | WAIT_MCU_DONE |
| WAIT_MCU_DONE | 收到 `D` | WAIT_RETURN_TO_PENDING |
| WAIT_MCU_DONE | 收到 `F` | FULL_PAUSED |
| WAIT_MCU_DONE | 超时或收到 `E` | ERROR_RECOVERY |
| WAIT_RETURN_TO_PENDING | 连续 ≥8 帧识别到「待分拣」 | IDLE_WAIT_VISUAL |
| WAIT_RETURN_TO_PENDING | 收到 `F` | FULL_PAUSED |
| IDLE_WAIT_VISUAL | 收到 `F` | FULL_PAUSED |
| CANDIDATE_DETECTED | 收到 `F` | FULL_PAUSED |
| FULL_PAUSED | 收到 `N` | IDLE_WAIT_VISUAL |
| ERROR_RECOVERY | 恢复策略完成 | IDLE_WAIT_VISUAL |

---

## 5. 与旧版状态机对比

| 旧版状态 (09/10) | 新版状态 | 变化说明 |
|------------------|----------|----------|
| WAIT_TRIGGER | **已删除** | 不再等待 MCU 发 T |
| PREDICT | IDLE_WAIT_VISUAL / CANDIDATE_DETECTED | 持续推理，不依赖外部触发 |
| — | WAIT_RETURN_TO_PENDING | **新增**：确保垃圾已清除才允许下一轮 |
| WAIT_DONE | WAIT_MCU_DONE | 保留，增加超时保护 |
| FULL | FULL_PAUSED | 保留 |
| — | ERROR_RECOVERY | **新增**：明确的错误恢复状态 |

---

## 6. 下一阶段实现要点

- 状态机逻辑将在 `rpi/` 目录中实现为 Python 类 `VisionTriggerStateMachine`
- 需要与 `runtime_config.example.json` 中的 decision 参数联动
- MCU 侧固件不需要实现状态机，MCU 只需响应单字符命令并返回 D/F/N/E
