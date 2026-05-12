$systems = @("long_context", "rag", "mem0", "memgpt", "zep", "a-mem")
$dataset = "extreme_integrity_dataset.json"
$outdir = "results_50_samples"

New-Item -ItemType Directory -Force -Path $outdir

foreach ($sys in $systems) {
    $output = "$outdir/extreme_results_50_$sys.csv"
    Write-Host "Starting evaluation for $sys..."
    python evaluate.py --system $sys --dataset $dataset --output $output --limit 50
}
