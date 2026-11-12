param([string]$In, [string]$OutDocx, [string]$OutPdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($In, $false, $false)
    $doc.Fields.Update() | Out-Null
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.UpdatePageNumbers() }
    $doc.SaveAs2($OutDocx, 16)          # wdFormatDocumentDefault
    $doc.ExportAsFixedFormat($OutPdf, 17) # wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)
    "words: " + $doc.ComputeStatistics(0)
    $doc.Close($false)
} finally { $word.Quit() }
