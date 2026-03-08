@echo off
:: 用法：双击这个文件，或者在终端输入 install_and_update.bat 包名
if "%1"=="" (
    echo 请输入要安装的包名，示例：install_and_update.bat faiss-cpu
    pause
    exit /b
)
pip install %1
pip freeze > requirements.txt
echo 包安装完成，requirements.txt已更新！
pause