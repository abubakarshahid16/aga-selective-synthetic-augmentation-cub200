$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = "python"
$commonArgs = @(
    "--python", $python,
    "--project-src", "src",
    "--prepared-dir", "data\prepared",
    "--image-root", "data\CUB_200_2011\images",
    "--model-name", "resnet18",
    "--epochs", "30",
    "--patience", "7",
    "--batch-size", "32",
    "--image-size", "224",
    "--num-workers", "0",
    "--seeds", "42", "43", "44",
    "--resume"
)

& $python "src\run_publication_benchmark.py" @commonArgs
& $python "src\ablation.py" @commonArgs --confidence-threshold "0.02" --clip-threshold "0.20"
& $python "src\publication_readiness.py"
