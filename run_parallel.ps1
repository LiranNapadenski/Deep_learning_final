$systems = @("rag", "mem0", "memgpt", "zep", "a-mem")
$dataset = "extreme_integrity_dataset.json"
$outdir = "results_50_samples"

# long_context is already running in the background, so I excluded it.

foreach ($sys in $systems) {
    $output = "$outdir/extreme_results_50_$sys.csv"
    Write-Host "Starting evaluation for $sys in background..."
    Start-Process -FilePath "python" -ArgumentList "evaluate.py --system $sys --dataset $dataset --output $output --limit 50" -WindowStyle Hidden
}
