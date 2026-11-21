<#
.SYNOPSIS
    Automated Windows Task Scheduler Setup for NeuroTrade Daily Orchestrator.

.DESCRIPTION
    Registers, unregisters, inspects, or manually triggers the automated daily
    execution task for the NeuroTrade algorithmic trading system.

    Scheduled execution:
    - Runs Monday through Friday at 16:30 IST (after NSE trading close and Bhavcopy release).
    - Invokes: neuro_env\Scripts\python.exe backend\live_data\daily_orchestrator.py
    - Includes holiday exclusion, duplicate run prevention, and process lock safeguards.

.PARAMETER Action
    Install, Uninstall, Status, or RunNow.

.EXAMPLE
    .\scripts\setup_windows_scheduler.ps1 -Action Install
    .\scripts\setup_windows_scheduler.ps1 -Action Status
    .\scripts\setup_windows_scheduler.ps1 -Action RunNow
    .\scripts\setup_windows_scheduler.ps1 -Action Uninstall
#>

param (
    [ValidateSet("Install", "Uninstall", "Status", "RunNow")]
    [string]$Action = "Status"
)

$TaskName = "NeuroTrade-DailyOrchestrator"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $ProjectRoot "neuro_env\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectRoot "backend\live_data\daily_orchestrator.py"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   NEUROTRADE — WINDOWS TASK SCHEDULER MANAGEMENT UTILITY" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Project Directory : $ProjectRoot"
Write-Host " Python Executable : $PythonExe"
Write-Host " Orchestrator Entry: $ScriptPath"
Write-Host " Scheduled Task    : $TaskName"
Write-Host ' Target Timing     : Monday - Friday at 16:30 IST (Post-NSE Close)'
Write-Host "-----------------------------------------------------------------"

if (-not (Test-Path $PythonExe)) {
    Write-Warning "Python environment not found at '$PythonExe'. Please verify path."
}
if (-not (Test-Path $ScriptPath)) {
    Write-Warning "Orchestrator script not found at '$ScriptPath'."
}

switch ($Action) {
    "Status" {
        Write-Host "`n[Action: Checking Task Status]" -ForegroundColor Yellow
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            $info = Get-ScheduledTaskInfo -TaskName $TaskName
            Write-Host "  -> Task Status       : $($task.State)" -ForegroundColor Green
            Write-Host "  -> Last Run Time     : $($info.LastRunTime)"
            Write-Host "  -> Last Task Result  : $($info.LastTaskResult)"
            Write-Host "  -> Next Run Time     : $($info.NextRunTime)"
            Write-Host "  -> Number of Missed  : $($info.NumberOfMissedRuns)"
        } else {
            Write-Host "  -> Task '$TaskName' is NOT currently installed in Windows Task Scheduler." -ForegroundColor DarkYellow
            Write-Host "     Run with '-Action Install' to register."
        }
    }

    "Install" {
        Write-Host "`n[Action: Installing Scheduled Task]" -ForegroundColor Yellow

        # Check if already installed
        $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Host "  -> Task '$TaskName' already exists. Unregistering previous version..."
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        }

        # Action: python.exe backend\live_data\daily_orchestrator.py
        $taskAction = New-ScheduledTaskAction `
            -Execute $PythonExe `
            -Argument "backend\live_data\daily_orchestrator.py" `
            -WorkingDirectory $ProjectRoot

        # Trigger: Daily Monday-Friday at 16:30
        $taskTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At '16:30'

        # Settings: Allow on battery, wake if needed, restart on failure, limit execution to 1 hour
        $taskSettings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
            -RestartCount 3 `
            -RestartInterval (New-TimeSpan -Minutes 5) `
            -StartWhenAvailable

        # Register task for current user
        try {
            Register-ScheduledTask `
                -TaskName $TaskName `
                -Action $taskAction `
                -Trigger $taskTrigger `
                -Settings $taskSettings `
                -Description "Automated Daily Market Ingestion, LSTM/PPO Inference, Paper Order Settlement, and Multi-Session Performance Evaluation for NeuroTrade." `
                -ErrorAction Stop | Out-Null

            Write-Host "  -> SUCCESS: Task '$TaskName' registered successfully!" -ForegroundColor Green
            Write-Host "  -> Scheduled to run weekdays (Mon-Fri) at 16:30 IST."
            Write-Host "  -> Verifying status..."
            & $PSCommandPath -Action Status
        } catch {
            Write-Host "  -> FAILED to register task: $_" -ForegroundColor Red
            Write-Host "  -> Note: Administrator privileges may be required to register tasks for certain accounts."
        }
    }

    "Uninstall" {
        Write-Host "`n[Action: Uninstalling Scheduled Task]" -ForegroundColor Yellow
        $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existing) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "  -> SUCCESS: Task '$TaskName' has been removed from Windows Task Scheduler." -ForegroundColor Green
        } else {
            Write-Host "  -> Task '$TaskName' was not found. Nothing to uninstall." -ForegroundColor DarkYellow
        }
    }

    "RunNow" {
        Write-Host "`n[Action: Triggering Task Run Immediately]" -ForegroundColor Yellow
        $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existing) {
            Start-ScheduledTask -TaskName $TaskName
            Write-Host "  -> SUCCESS: Task triggered in background. Check reports\logs\ for execution logs." -ForegroundColor Green
        } else {
            Write-Host "  -> Task '$TaskName' is not installed. Running python directly in console instead..." -ForegroundColor Yellow
            & $PythonExe $ScriptPath --force
        }
    }
}

Write-Host "=================================================================" -ForegroundColor Cyan
