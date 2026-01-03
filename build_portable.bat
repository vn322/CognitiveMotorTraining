@echo off
echo 📦 Сборка ПЕРЕНОСИМОГО main.exe (всё в одном файле)...

:: Проверка наличия pose_landmark_cpu.binarypb в корне
if not exist "pose_landmark_cpu.binarypb" (
    echo ❌ Ошибка: pose_landmark_cpu.binarypb не найден в корне проекта.
    echo Скопируйте его из:
    echo   %USERPROFILE%\AppData\Local\Programs\Python\Python312\Lib\site-packages\mediapipe\modules\pose_landmark\
    pause
    exit /b 1
)

:: Проверка шрифта
if not exist "DejaVuSans.ttf" (
    echo ❌ Ошибка: DejaVuSans.ttf не найден.
    echo Скачайте: https://github.com/googlefonts/dejavu-sans-mono-fonts/raw/main/fonts/ttf/DejaVuSans.ttf
    pause
    exit /b 1
)

:: Сборка
pyinstaller --onefile --windowed ^
  --add-binary "pose_landmark_cpu.binarypb;mediapipe/modules/pose_landmark" ^
  --add-data "DejaVuSans.ttf;." ^
  --add-data "blocks;blocks" ^
  --add-data "config;config" ^
  --add-data "utils;utils" ^ 
  --collect-data cv2 ^
  --hidden-import reportlab.pdfbase.pdfmetrics ^
  --hidden-import reportlab.pdfbase.ttfonts ^
  --hidden-import pandas ^
  --hidden-import numpy.core._methods ^
  --hidden-import numpy.lib.format ^
  --icon=icon.ico ^
  --name "CognitiveMotorTraining" ^
  main.py

:: Готово
echo.
if exist "dist\main.exe" (
    echo ✅ УСПЕХ: dist\main.exe создан!
    echo 📁 Размер: %~z1 байт
    echo 🚀 Переносите ПАПКУ dist\ на любой Windows-ПК и запускайте main.exe
) else (
    echo ❌ ОШИБКА: main.exe не создан.
    echo Проверьте лог выше.
)

pause