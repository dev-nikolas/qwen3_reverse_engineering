@echo off
setlocal
echo ====================================================================
echo   QWEN3-1.7B -- DESCARGA AUTOMATIZADA DE PESOS OFICIALES
echo ====================================================================
echo.

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3 tools\download_weights.py %*
    goto end
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python tools\download_weights.py %*
    goto end
)

echo [!] Error: No se encontro Python en el PATH. Instala Python 3.10+ para continuar.
pause
exit /b 1

:end
if %ERRORLEVEL% equ 0 (
    echo.
    echo [✓] Proceso finalizado correctamente.
) else (
    echo.
    echo [!] Hubo un problema durante la descarga.
)
pause
