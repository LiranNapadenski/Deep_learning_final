$systems = @("long_context", "rag", "mem0", "memgpt", "zep")
$dataset = "extreme_integrity_dataset.json"

foreach ($sys in $systems) {
    $output = "extreme_results_$sys.csv"
    Write-Host "Starting evaluation for $sys..."
    python evaluate.py --system $sys --dataset $dataset --output $output --limit 100
}
