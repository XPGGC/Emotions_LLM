@echo off
REM 启动 Flask 后端（指定 Anaconda 环境 python 路径）
start cmd /k "cd /d %~dp0 && D:\Anaconda3\envs\wzry\python.exe app.py"

REM 启动 React 前端
start cmd /k "cd /d %~dp0\face-recognition-app && npm start"

pause