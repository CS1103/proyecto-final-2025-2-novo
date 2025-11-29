@echo off
echo ================================
echo Running Neural Network Tests
echo ================================
echo.

set passed=0
set failed=0

echo Running test_matrix...
if exist build\test_matrix.exe (
    build\test_matrix.exe
    if %errorlevel% equ 0 (
        echo [PASS] test_matrix
        set /a passed+=1
    ) else (
        echo [FAIL] test_matrix
        set /a failed+=1
    )
) else (
    echo [FAIL] test_matrix not found
    set /a failed+=1
)
echo.

echo Running test_dense...
if exist build\test_dense.exe (
    build\test_dense.exe
    if %errorlevel% equ 0 (
        echo [PASS] test_dense
        set /a passed+=1
    ) else (
        echo [FAIL] test_dense
        set /a failed+=1
    )
) else (
    echo [FAIL] test_dense not found
    set /a failed+=1
)
echo.

echo Running test_activation...
if exist build\test_activation.exe (
    build\test_activation.exe
    if %errorlevel% equ 0 (
        echo [PASS] test_activation
        set /a passed+=1
    ) else (
        echo [FAIL] test_activation
        set /a failed+=1
    )
) else (
    echo [FAIL] test_activation not found
    set /a failed+=1
)
echo.

echo Running test_xor...
if exist build\test_xor.exe (
    build\test_xor.exe
    if %errorlevel% equ 0 (
        echo [PASS] test_xor
        set /a passed+=1
    ) else (
        echo [FAIL] test_xor
        set /a failed+=1
    )
) else (
    echo [FAIL] test_xor not found
    set /a failed+=1
)
echo.

echo ================================
echo Test Summary
echo ================================
echo Passed: %passed%
echo Failed: %failed%
echo ================================

if %failed% equ 0 (
    echo All tests passed!
    exit /b 0
) else (
    echo Some tests failed.
    exit /b 1
)
