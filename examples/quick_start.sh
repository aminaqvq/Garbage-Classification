#!/bin/bash
echo "========== 垃圾分类分拣系统 =========="
echo "1. AI 识别预览"
echo "2. RKHO 串口手动测试"
echo "3. 最终完整系统（超声波+满载+AI+RKHO）"
echo "4. 旧 AA55 协议参考"
read -p "选择 (1-4): " c
D="$(dirname "$0")/../Garbage_Classification_Organized"
case $c in
  1) cd "$D/05_RPi_AI_Preview" && python3 rpi_ai_preview_camera.py ;;
  2) cd "$D/06_RKHO_Serial_Protocol_Test" && python3 rpi_manual_rkho_protocol_test.py ;;
  3) cd "$D/10_Final_Integrated_System" && python3 rpi_final_ai_sorting_with_full_load.py ;;
  4) cd "$D/08_AA55_Ultrasonic_Protocol_Reference" && python3 rpi_aa55_serial_handshake_test_old.py ;;
  *) echo "无效" ;;
esac
