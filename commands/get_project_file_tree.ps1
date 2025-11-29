function Show-Tree {
    param(
        [string]$Path = ".",
        [int]$Indent = 0
    )

    # Get all items in the current path, excluding .venv
    $items = Get-ChildItem -LiteralPath $Path | Where-Object { $_.Name -ne ".venv" }

    foreach ($item in $items) {
        # Print indentation + item name
        Write-Output (" " * $Indent + "|-- " + $item.Name)

        # If it's a directory, recurse
        if ($item.PSIsContainer) {
            Show-Tree -Path $item.FullName -Indent ($Indent + 4)
        }
    }
}

# Run it on the current directory and redirect the output to structure.txt
Show-Tree > structure.txt