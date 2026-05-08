@echo off
cd /d C:\SVC_INSPECAO_MOLAS
call .venv_svc\Scripts\activate.bat
python -m streamlit run app_camera_infer_dual_freeze_v19_final_v8_1_stable_reset.py
pause
