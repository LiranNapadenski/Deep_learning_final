# This script sets up both MemGPT and Zep backend servers locally.

Write-Host "Checking for Docker required by Zep..."
docker --version

Write-Host "Setting up Zep..."
# Download Zep's docker compose
if (-Not (Test-Path -Path .\zep-quickstart) ) {
    New-Item -ItemType Directory -Force -Path .\zep-quickstart
}
Set-Location -Path .\zep-quickstart
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/getzep/zep/main/docker-compose.yaml" -OutFile "docker-compose.yaml"
# Spin up Zep in detached mode
docker-compose up -d
Set-Location -Path ..

Write-Host "Setting up MemGPT..."
# It is assumed `pymemgpt` and `google-generativeai` are installed from requirements.txt
Write-Host "You must execute: memgpt server"
Write-Host "In a separate console tab to run the API at http://localhost:8283"
