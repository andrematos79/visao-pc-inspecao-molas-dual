# Executar dentro de C:\SVC_INSPECAO_MOLAS no PC de produção

# 1) Ativar ambiente virtual
.\.venv_svc\Scripts\Activate.ps1

# 2) Gerar lock file exato do ambiente que está FUNCIONANDO
pip freeze | Out-File -Encoding utf8 requirements_lock_prod.txt

# 3) Opcional: salvar lista resumida de pacotes top-level
pip list --format=freeze | Out-File -Encoding utf8 requirements_full_list.txt

# 4) Conferência rápida
Get-Content .\requirements_lock_prod.txt | Select-Object -First 40
