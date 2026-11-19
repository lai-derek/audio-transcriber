param([Parameter(Mandatory=$true)][int]$ProcessId)
. "$PSScriptRoot/WindowsUiAutomation.ps1"
$process = Get-Process -Id $ProcessId -ErrorAction Stop
# Retain a handle before exit so Windows PowerShell can read a foreign process's exit code.
$null = $process.Handle
$window = WaitWindow $ProcessId 'Audio Transcriber — Lecture library'
$handle = [OwnWindow]::Find($ProcessId)
[OwnWindow]::PostMessage($handle,0x10,[IntPtr]::Zero,[IntPtr]::Zero) | Out-Null
if (-not $process.WaitForExit(20000)) { throw 'Updated application did not close gracefully.' }
if ($null -eq $process.ExitCode -or $process.ExitCode -ne 0) { throw "Updated application exit code: $($process.ExitCode)" }
'Updater reopened the installed application and it closed gracefully.'
