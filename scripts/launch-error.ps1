# Shows why OpenFlo could not start, in a small window whose text can be
# selected and copied (a plain message box's text cannot), and saves the
# same text to %TEMP%\openflo-launch-error.txt. Web links and the folder
# and file paths in it are blue and underlined and open when clicked; an
# Open folder button opens the OpenFlo folder. openflo-gui.bat sets
# OPENFLO_MSG and OPENFLO_ROOT before calling it.
$ErrorActionPreference = 'Stop'
$nl = [Environment]::NewLine
$file = Join-Path $env:TEMP 'openflo-launch-error.txt'
$msg = $env:OPENFLO_MSG + $nl + $nl
if ($env:OPENFLO_ROOT) { $msg += 'OpenFlo folder: ' + $env:OPENFLO_ROOT.TrimEnd('\') + $nl }
$msg += 'Report the problem: https://www.getopenflo.com/feedback'
Set-Content -LiteralPath $file -Value $msg -Encoding UTF8
$msg += $nl + 'This text is also saved in: ' + $file
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object System.Windows.Forms.Form
$form.Text = 'OpenFlo could not start'
$form.Size = New-Object System.Drawing.Size(600, 300)
$form.StartPosition = 'CenterScreen'
$form.TopMost = $true
$box = New-Object System.Windows.Forms.RichTextBox
$box.Multiline = $true
$box.ReadOnly = $true
$box.DetectUrls = $false
$box.ScrollBars = 'Vertical'
$box.Dock = 'Fill'
# The box counts each line break as one character, so the text it holds
# uses "`n" line breaks, and each link is located with the box's own Find:
# offsets taken from a "`r`n" string drift by one per line.
$box.Text = $msg -replace "`r`n", "`n"
$links = @()
$pattern = 'https?://[^\s<>"]+[^\s<>".,;:)]|[A-Za-z]:\\[^\r\n]+'
$from = 0
foreach ($m in [regex]::Matches($box.Text, $pattern)) {
    $target = $m.Value.TrimEnd()
    $at = $box.Find($target, $from, [System.Windows.Forms.RichTextBoxFinds]::MatchCase)
    if ($at -lt 0) { continue }
    $from = $at + $target.Length
    $links += [pscustomobject]@{ Start = $at; End = $at + $target.Length; Target = $target }
    $box.Select($at, $target.Length)
    $box.SelectionColor = [System.Drawing.Color]::FromArgb(11, 92, 173)
    $box.SelectionFont = New-Object System.Drawing.Font($box.Font, [System.Drawing.FontStyle]::Underline)
}
$box.Select(0, 0)
function Get-LinkAt($point) {
    $i = $box.GetCharIndexFromPosition($point)
    foreach ($l in $links) { if ($i -ge $l.Start -and $i -lt $l.End) { return $l } }
    return $null
}
function Open-Link($target) {
    if ($target -match '^https?://') { Start-Process $target }
    elseif (Test-Path -LiteralPath $target -PathType Leaf) { Start-Process explorer.exe -ArgumentList "/select,`"$target`"" }
    else { Start-Process explorer.exe -ArgumentList "`"$target`"" }
}
$box.Add_MouseMove({ param($s, $e)
    if (Get-LinkAt $e.Location) { $box.Cursor = [System.Windows.Forms.Cursors]::Hand }
    else { $box.Cursor = [System.Windows.Forms.Cursors]::IBeam } })
$box.Add_MouseClick({ param($s, $e)
    $l = Get-LinkAt $e.Location
    if ($l -and $box.SelectionLength -eq 0) { Open-Link $l.Target } })
$buttons = New-Object System.Windows.Forms.FlowLayoutPanel
$buttons.FlowDirection = 'RightToLeft'
$buttons.Dock = 'Bottom'
$buttons.Height = 40
$ok = New-Object System.Windows.Forms.Button
$ok.Text = 'OK'
$ok.Add_Click({ $form.Close() })
$copy = New-Object System.Windows.Forms.Button
$copy.Text = 'Copy'
$copy.Add_Click({ [System.Windows.Forms.Clipboard]::SetText($box.Text) })
$buttons.Controls.Add($ok)
$buttons.Controls.Add($copy)
if ($env:OPENFLO_ROOT -and (Test-Path -LiteralPath $env:OPENFLO_ROOT)) {
    $open = New-Object System.Windows.Forms.Button
    $open.Text = 'Open folder'
    $open.AutoSize = $true
    $open.Add_Click({ Start-Process explorer.exe -ArgumentList "`"$($env:OPENFLO_ROOT.TrimEnd('\'))`"" })
    $buttons.Controls.Add($open)
}
$form.Controls.Add($box)
$form.Controls.Add($buttons)
$form.AcceptButton = $ok
[void]$form.ShowDialog()
