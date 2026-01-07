#!/bin/bash
# 简单的可视化启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HTML_FILE="${SCRIPT_DIR}/render_trajectories.html"

# 检查HTML文件是否存在
if [ ! -f "$HTML_FILE" ]; then
    echo "错误: 找不到 render_trajectories.html 文件"
    exit 1
fi

# 允许通过命令行参数指定端口，默认使用8001（避免与8000冲突）
PORT=${1:-8001}

# 尝试找到可用端口
MAX_ATTEMPTS=10
ATTEMPT=0
ORIGINAL_PORT=$PORT

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # 检查端口是否被占用（使用 netstat 或 ss 命令）
    if command -v netstat >/dev/null 2>&1; then
        if ! netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ! ss -tuln 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
    else
        # 如果 netstat 和 ss 都不可用，尝试直接启动，如果失败再换端口
        break
    fi
    
    if [ $ATTEMPT -eq 0 ] && [ "$1" = "" ]; then
        echo "端口 $PORT 已被占用，尝试其他端口..."
    fi
    PORT=$((PORT + 1))
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $PORT -ne $ORIGINAL_PORT ] && [ "$1" = "" ]; then
    echo "使用端口: $PORT (原端口 $ORIGINAL_PORT 被占用)"
fi

# 启动简单的HTTP服务器
echo "正在启动可视化服务器..."
echo "请在浏览器中打开: http://localhost:$PORT/render_trajectories.html"
echo "按 Ctrl+C 停止服务器"
echo ""

cd "$SCRIPT_DIR"
python3 -m http.server $PORT 2>&1

