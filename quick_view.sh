#!/bin/bash
# 快速启动可视化服务器（使用8001端口）

cd "$(dirname "$0")"
PORT=8001

echo "正在启动可视化服务器..."
echo "请在浏览器中打开: http://localhost:$PORT/render_trajectories.html"
echo "按 Ctrl+C 停止服务器"
echo ""

python3 -m http.server $PORT

