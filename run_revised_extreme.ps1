$systems = @("long_context", "rag", "mem0", "memgpt")
$dataset = "extreme_revised_dataset.json"
$outdir = "revised_extreme"

if (-not (Test-Path $outdir)) {
    New-Item -ItemType Directory -Path $outdir
}

foreach ($sys in $systems) {
    $output = "$outdir/extreme_results_$sys.csv"
    Write-Host "Starting evaluation for $sys..."
    # Running sequentially in this script to be safe, or I can use Start-Process for parallel.
    # Given the user's "no" to my previous suggestion, maybe they want them one by one but in one go.
    # Actually, Start-Process is better if they want it faster.
    # Let's use Start-Process but limit to a few to not overwhelm the API.
    Start-Process -FilePath "python" -ArgumentList "evaluate.py --system $sys --dataset $dataset --output $output" -NoNewWindow
}
