@echo off
chcp 65001 >nul
title AI情感治疗系统

echo.
echo ========================================
echo        AI情感治疗系统启动器
echo ========================================
echo.

echo 正在加载环境变量配置...
if exist config.env (
    for /f "tokens=1,2 delims==" %%a in (config.env) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set %%a=%%b
            echo 已设置: %%a
        )
    )
    echo 环境变量配置已加载
) else (
    echo 警告: 未找到 config.env 文件，将使用默认配置
)

echo.
echo 正在启动AI情感治疗系统...

echo 启动后端服务...
start "AI情感治疗系统后端" cmd /k "D:\Miniconda3\envs\emotions-llm\python.exe app.py"

timeout /t 3 /nobreak >nul

echo 启动前端界面...
cd face-recognition-app
start "AI情感治疗系统前端" cmd /k "npm start"

echo.
echo ========================================
echo        系统启动完成！
echo ========================================
echo.
echo 前端界面: http://localhost:3000
echo 后端API:  http://127.0.0.1:5000
echo.
echo 注意: 后端和前端服务已在独立窗口中启动
echo 关闭系统时请分别关闭这两个窗口
echo.
pause 