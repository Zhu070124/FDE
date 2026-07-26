@echo off
chcp 65001 >nul
title CQUPT AI 学生成长助手

echo ========================================
echo   🎓 CQUPT AI 学生成长助手
echo   基于RAG的一站式咨询平台
echo ========================================
echo.

cd /d "%~dp0backend"

echo [1/4] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
echo ✅ Python 就绪

echo [2/4] 安装依赖...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ⚠️ 部分依赖安装失败，尝试继续...
)
echo ✅ 依赖就绪

echo [3/4] 检查豆包API Key...
if "%DOUBAO_API_KEY%"=="" (
    echo ⚠️ 未设置 DOUBAO_API_KEY 环境变量
    echo 请在 config.py 中手动设置或运行: set DOUBAO_API_KEY=你的key
)

echo [4/4] 启动服务...
echo.
echo 🚀 服务启动: http://localhost:8000
echo 📖 API文档: http://localhost:8000/docs
echo.
python main.py

pause
