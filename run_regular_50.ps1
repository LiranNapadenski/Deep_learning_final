$systems = @("long_context", "rag", "mem0", "memgpt", "zep", "a-mem")
$dataset = "conflict_interference_dataset.json"
$outdir = "results_50_regular"

New-Item -ItemType Directory -Force -Path $outdir | Out-Null

foreach ($sys in $systems) {
    $output = "$outdir/regular_results_50_$sys.csv"
    Write-Host "Starting evaluation for $sys in background..."
    Start-Process -FilePath "python" -ArgumentList "evaluate.py --system $sys --dataset $dataset --output $output --limit 50" -WindowStyle Hidden
}
Write-Host "All background jobs started."
