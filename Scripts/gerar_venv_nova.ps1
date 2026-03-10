Set-Location "C:\SVC_INSPECAO_MOLAS"

if (Test-Path ".venv_svc") {
    Write-Host "Ambiente .venv_svc já existe."
} else {
    python -m venv .venv_svc
    Write-Host "Ambiente virtual criado com sucesso."
}