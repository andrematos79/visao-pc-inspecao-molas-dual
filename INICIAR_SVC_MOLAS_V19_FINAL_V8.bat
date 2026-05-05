@echo off
cd /d C:\SVC_INSPECAO_MOLAS
if exist .venv_svc\Scripts\activate.bat (
    call .venv_svc\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python -m streamlit run app_camera_infer_dual_freeze_v19_final_v8.py
pause
