# Запуск SearXNG для DIBot.
# Поднимает Docker Desktop при необходимости, ждёт движок и запускает контейнер.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Docker {
    docker info *> $null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Docker)) {
    Write-Host "Docker не запущен, поднимаю Docker Desktop…"
    $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $desktop)) {
        Write-Error "Docker Desktop не найден: $desktop"
        exit 1
    }
    Start-Process $desktop
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        if (Test-Docker) { break }
        Write-Host "  ожидание движка…"
    }
    if (-not (Test-Docker)) {
        Write-Error "Движок Docker не поднялся за 3 минуты"
        exit 1
    }
}

Write-Host "Проверяю конфигурацию…"
docker compose config *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker-compose.yml содержит ошибку"
    exit 1
}

Write-Host "Запускаю SearXNG…"
docker compose up -d

Write-Host "Ожидание готовности…"
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest "http://localhost:8080/healthz" -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) { break }
    } catch { }
    Start-Sleep -Seconds 3
}

Write-Host "Проверяю JSON-формат…"
try {
    $probe = Invoke-RestMethod "http://localhost:8080/search?q=test&format=json" -TimeoutSec 10
    if ($probe.results) {
        Write-Host "SearXNG готов: http://localhost:8080 (JSON включён, результатов: $($probe.results.Count))"
    } else {
        Write-Host "SearXNG отвечает, но выдача пустая — проверь engines в docker/searxng/settings.yml"
    }
} catch {
    Write-Warning "JSON-формат не отвечает: $($_.Exception.Message)"
    Write-Host "Проверь docker\searxng\settings.yml -> search.formats (должен быть json)"
}

Write-Host ""
Write-Host "Остановить:  docker compose down"
