# run.ps1 — wrapper to activate sdsg_sim conda env and run a Python script
# Usage: .\run.ps1 phantom\navy_phantom.py
#        .\run.ps1 -m pytest tests\

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$base = "$env:USERPROFILE\miniforge3\envs\sdsg_sim"
$env:PATH = "$base;" +
            "$base\Library\mingw-w64\bin;" +
            "$base\Library\usr\bin;" +
            "$base\Library\bin;" +
            "$base\Scripts;" +
            $env:PATH

& "$base\python.exe" @Args
