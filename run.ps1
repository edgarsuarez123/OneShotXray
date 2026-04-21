# run.ps1 — wrapper to activate sdsg_sim conda env and run a Python script
# Usage: .\run.ps1 phantom\navy_phantom.py
#        .\run.ps1 -m pytest tests\

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$env:PATH = "C:\Users\Edgar\miniforge3\envs\sdsg_sim;" +
            "C:\Users\Edgar\miniforge3\envs\sdsg_sim\Library\mingw-w64\bin;" +
            "C:\Users\Edgar\miniforge3\envs\sdsg_sim\Library\usr\bin;" +
            "C:\Users\Edgar\miniforge3\envs\sdsg_sim\Library\bin;" +
            "C:\Users\Edgar\miniforge3\envs\sdsg_sim\Scripts;" +
            $env:PATH

& "C:\Users\Edgar\miniforge3\envs\sdsg_sim\python.exe" @Args
