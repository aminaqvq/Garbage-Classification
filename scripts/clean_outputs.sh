#!/bin/bash
# 清理输出目录
# 用法：bash scripts/clean_outputs.sh

set -e

echo "========== 清理输出文件 =========="

# 清理训练输出
if [ -d "outputs" ]; then
    echo "删除 outputs/ ..."
    rm -rf outputs/*
    echo "  done"
fi

# 清理导出模型
if [ -d "export" ]; then
    echo "删除 export/ 中的模型文件 ..."
    rm -rf export/quant_run_* export/latest_*
    echo "  done"
fi

# 清理日志
if [ -d "Logs" ]; then
    echo "删除 Logs/ ..."
    rm -rf Logs/*
    echo "  done"
fi

# 清理截图
for dir in Captures_Final Captures_Locked_Trigger Captures_Servo_Char Captures_AI_Test; do
    if [ -d "$dir" ]; then
        echo "清空 $dir/ ..."
        rm -rf "$dir"/*
        echo "  done"
    fi
done

# 清理 Python 缓存
echo "清理 __pycache__ ..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  done"

echo ""
echo "========== 清理完成 =========="
