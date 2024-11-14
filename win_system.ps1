while (1) {
    # CPU core
    $cpuUsage = (Get-Counter '\Processor Information(*)\% Processor Time').CounterSamples
    $cpuText = " CPU usage: "
    foreach ($cpu in $cpuUsage) {
        $cpuText += "[$($cpu.InstanceName)] $([math]::Round($cpu.CookedValue, 0))% "
    }

    # Check memory usage
    $totalMemory = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1MB
    $availableMemory = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
    $usedMemory = $totalMemory - $availableMemory

    $memoryText = "    Memory: $([math]::Round($usedMemory / 1024, 2))/$([math]::Round($totalMemory / 1024, 2)) GB"
    
    # Clear-Host

    # GPU
    Write-Host ""
    gpustat -a --no-processes  #gpustat -a --no-processes -i 1
    Write-Host $cpuText -NoNewline
    Write-Host $memoryText -NoNewline

    # sleep 1
}





