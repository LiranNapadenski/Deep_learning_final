# Run this script in an ADMINISTRATOR PowerShell window

Write-Host "Downloading Microsoft Visual C++ Build Tools (this may take a moment)..."
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile "vs_buildtools.exe"

Write-Host "Installing C++ Build Tools (this will download ~2GB and take 5-15 minutes)..."
Write-Host "A User Account Control (UAC) prompt may appear. Please click 'Yes' to allow the installation."

# Start the installer silently but wait for completion
$arguments = "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
Start-Process -FilePath ".\vs_buildtools.exe" -ArgumentList $arguments -Wait

Write-Host "Installation complete! You can now safely delete vs_buildtools.exe."
Write-Host "You should now be able to run 'pip install mem0ai' without the chroma-hnswlib error!"
