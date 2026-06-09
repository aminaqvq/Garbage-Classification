#!/bin/bash
# 快速启动脚本
# 用法：bash examples/quick_start.sh

echo "========== 垃圾分类分拣系统 =========="
echo ""
echo "请选择功能："
echo "1. AI 识别预览（仅摄像头，不控制硬件）"
echo "2. 串口通信测试（需要连接 52RC）"
echo "3. 完整分拣系统（需要连接 52RC + 舵机 + 电机）"
echo "4. 单字符舵机控制（需要连接 52RC）"
echo "5. 锁定触发版（需要连接 52RC + 超声波）"
echo ""
read -p "请输入编号 (1-5): " choice

case $choice in
    1) cd rpi && python3 ai_preview.py ;;
    2) cd rpi && python3 serial_handshake_test.py ;;
    3) cd rpi && python3 final_sorting_system.py ;;
    4) cd rpi && python3 servo_char_control.py ;;
    5) cd rpi && python3 locked_trigger_system.py ;;
    *) echo "无效选择" ;;
esac
