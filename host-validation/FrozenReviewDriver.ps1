# Internal driver for Test-FrozenReview.py. Save with a UTF-8 BOM for Windows PowerShell 5.
param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$true)][string]$Data,
    [Parameter(Mandatory=$true)][string]$Report
)
. "$PSScriptRoot/WindowsUiAutomation.ps1"
$button=[System.Windows.Automation.ControlType]::Button
$edit=[System.Windows.Automation.ControlType]::Edit
$combo=[System.Windows.Automation.ControlType]::ComboBox
$runs=@()
$playback=$null
$env:QT_QPA_PLATFORM='windows'
$lecture=@(Get-ChildItem -LiteralPath (Join-Path $Data 'lectures') -Directory)[0].FullName
$raw=Get-Content -LiteralPath (Join-Path $lecture 'raw-transcript.json') -Raw | ConvertFrom-Json
$starts=@($raw.segments | ForEach-Object { [double]$_.start })
function Stamp([double]$Seconds) { $s=[int][math]::Floor($Seconds); return ('{0}:{1:D2}' -f [int][math]::Floor($s/60), ($s%60)) }
function PostKey([IntPtr]$Handle,$Element,[int]$Key) {
    # Key messages reach the focused widget of the window that receives them.
    $Element.SetFocus()
    Start-Sleep -Milliseconds 100
    [OwnWindow]::PostMessage($Handle,0x100,[IntPtr]$Key,[IntPtr]1) | Out-Null
    [OwnWindow]::PostMessage($Handle,0x101,[IntPtr]$Key,[IntPtr]0xC0000001) | Out-Null
}
function SelectTab([IntPtr]$Handle,$Root,[string]$Name) {
    ClickElement $Handle (FindControl $Root $Name ([System.Windows.Automation.ControlType]::TabItem))
    Start-Sleep -Milliseconds 300
}
function EditCount { return @(Get-ChildItem -LiteralPath (Join-Path $lecture 'edits') -Filter '*.json' -ErrorAction SilentlyContinue).Count }
function TypeCorrection($Transcript,[string]$Text) {
    # The first paragraph is "<time>   <words>"; replace its words, keep everything else.
    $value=(ValueOf $Transcript).Current.Value
    $lines=$value -split "`n"
    $lines[0]=(Stamp $starts[0])+'   '+$Text
    $before=EditCount
    (ValueOf $Transcript).SetValue(($lines -join "`n"))
    WaitUntil { (EditCount) -gt $before } 'Typed correction was not saved as an Edited revision'
    WaitUntil { (ValueOf $Transcript).Current.Value.Contains($Text) } 'Saved text absent from the visible transcript'
}
for($attempt=0; $attempt -lt 2; $attempt++) {
    $p=Start-Process -FilePath $Exe -ArgumentList @('--data-dir',('"'+$Data+'"')) -WorkingDirectory $Data -PassThru -WindowStyle Hidden
    $h=[IntPtr]::Zero
    try {
        $main=WaitWindow $p.Id 'Audio Transcriber — Lecture library'
        $h=[OwnWindow]::Find($p.Id)
        [OwnWindow]::ShowWindow($h,5) | Out-Null
        if(-not (Test-Path -LiteralPath (Join-Path $Data 'desktop.lock'))) { throw 'Library lock absent' }
        $transcript=FindControl $main 'Transcript' $edit
        if($attempt -eq 0) {
            # Original opens editable: typing in it is how corrections are made.
            if((ValueOf $transcript).Current.IsReadOnly) { throw 'Original transcript is not editable' }
            InvokeControl (FindControl $main 'Live dictation…' $button)
            $dictation=WaitWindow $p.Id 'Audio Transcriber — Live dictation'
            $dictationHandle=[OwnWindow]::Find($p.Id,'Audio Transcriber — Live dictation')
            FindControl $dictation 'Microphone' $combo | Out-Null
            FindControl $dictation 'Record a copy' $button | Out-Null
            if((ValueOf (FindControl $dictation 'Cleanup' $combo)).Current.Value -ne 'Off') { throw 'Fresh dictation cleanup default differs' }
            FindControl $dictation 'Mostly prose' ([System.Windows.Automation.ControlType]::Text) | Out-Null
            foreach($device in @('GPU','NPU','CPU (recommended)')) {
                SelectCombo $p.Id $dictationHandle $dictation 'Speech device' $device
            }
            $shortcut=ValueOf (FindControl $dictation 'Dictation shortcut' $edit)
            $shortcut.SetValue('Ctrl+Shift+F10')
            if($shortcut.Current.Value -ne 'Ctrl+Shift+F10') { throw 'Frozen dictation shortcut was not editable' }
            foreach($mode in @('Hold to talk','Toggle')) {
                SelectCombo $p.Id $dictationHandle $dictation 'Shortcut mode' $mode
            }
            [OwnWindow]::PostMessage($dictationHandle,0x10,[IntPtr]::Zero,[IntPtr]::Zero) | Out-Null
            WaitUntil { [OwnWindow]::FindClass($p.Id,'QDialog','Audio Transcriber — Live dictation') -eq [IntPtr]::Zero } 'Idle dictation window did not close'
            $slider=FindControl $main 'Audio position' ([System.Windows.Automation.ControlType]::Slider)
            $range=[System.Windows.Automation.RangeValuePattern]$slider.GetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern)
            $deadline=(Get-Date).AddSeconds(10)
            while($range.Current.Maximum -lt 61000 -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
            if($range.Current.Maximum -ne 61000) { throw 'Synthetic playback duration differs' }
            $range.SetValue(5000)
            Start-Sleep -Milliseconds 200
            $clock=@($main.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition) | Where-Object {$_.Current.Name -eq '00:00:05 / 00:01:01'})
            if($clock.Count -ne 1) { throw 'Slider value did not seek the actual player' }
            InvokeControl (FindControl $main 'Play' $button)
            Start-Sleep -Milliseconds 1500
            InvokeControl (FindControl $main 'Pause audio' $button)
            $advanced=$range.Current.Value
            if($advanced -lt 5500) { throw 'Playback failed to advance from the seek position' }
            # Clicking a line moves the audio to that line's start.
            $text=[System.Windows.Automation.TextPattern]$transcript.GetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern)
            # Qt's text ranges have no FindText; reach the line by character offset.
            $offset=(ValueOf $transcript).Current.Value.IndexOf((Stamp $starts[1]))
            if($offset -lt 0) { throw 'Second line time stamp not found in the transcript' }
            $start=[System.Windows.Automation.Text.TextPatternRangeEndpoint]::Start
            $end=[System.Windows.Automation.Text.TextPatternRangeEndpoint]::End
            $character=[System.Windows.Automation.Text.TextUnit]::Character
            $line=$text.DocumentRange.Clone()
            $line.MoveEndpointByRange($end,$line,$start)
            $line.Move($character,$offset) | Out-Null
            $line.MoveEndpointByUnit($end,$character,4) | Out-Null
            $boxes=@($line.GetBoundingRectangles())
            if($boxes.Count -gt 0 -and $boxes[0].Width -gt 0) {
                $box=$boxes[0]
                [OwnWindow]::Click($h,($box.Left+$box.Width/2),($box.Top+$box.Height/2))
            } else {
                $line.Select()  # Moving the cursor onto the line seeks the same way.
            }
            $expected=[int][math]::Round($starts[1]*1000)
            WaitUntil { [math]::Abs($range.Current.Value-$expected) -le 100 } 'Clicking a line did not move the audio to it'
            $playback=[ordered]@{requested_ms=5000; displayed_position_matches=$true; advanced_ms=$advanced; line_click_seek_ms=$range.Current.Value}
            foreach($correction in @('Synthetic native correction one','Synthetic native correction two')) {
                TypeCorrection $transcript $correction
            }
            $search=FindControl $main 'Find in transcript' $edit
            (ValueOf $search).SetValue('correction two')
            PostKey $h $search 0x0D
            Start-Sleep -Milliseconds 300
            $text=[System.Windows.Automation.TextPattern]$transcript.GetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern)
            $selection=@($text.GetSelection())
            if($selection.Count -ne 1 -or $selection[0].GetText(-1) -ne 'correction two') { throw 'Search did not select the edited phrase' }
        }
        $opening=[OwnWindow]::InvokeAsync((FindControl $main 'Revision history…' $button))
        $history=WaitWindow $p.Id 'Edited transcript history'
        Start-Sleep -Milliseconds 200
        $preview=ValueOf (FindControl $history 'Revision transcript' $edit)
        if(-not $preview.Current.IsReadOnly -or -not $preview.Current.Value.Contains('Synthetic native correction two')) { throw 'Newest revision missing or history editable' }
        $list=FindControl $history 'Saved revisions' ([System.Windows.Automation.ControlType]::List)
        $entries=$list.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty,[System.Windows.Automation.ControlType]::ListItem))
        if($entries.Count -ne 2) { throw 'Expected two saved revisions' }
        ClickElement ([OwnWindow]::Find($p.Id,'Edited transcript history')) $entries[1]
        WaitUntil { $preview.Current.Value.Contains('Synthetic native correction one') -and -not $preview.Current.Value.Contains('correction two') } 'Older revision preview did not update'
        InvokeControl (FindControl $history 'Close' $button)
        if(-not $opening.Wait(5000)) { throw 'History dialog did not close' }
        $deadline=(Get-Date).AddSeconds(5)
        while([OwnWindow]::Find($p.Id,'Edited transcript history') -ne [IntPtr]::Zero -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
        if([OwnWindow]::Find($p.Id,'Edited transcript history') -ne [IntPtr]::Zero) { throw 'History window remained after Close' }
        Start-Sleep -Milliseconds 200
        if($attempt -eq 0) {
            $exports=Join-Path (Split-Path $Data -Parent) 'exports'
            New-Item -ItemType Directory -Path $exports | Out-Null
            foreach($tab in @(@{name='Original';file='raw'},@{name='Edited';file='edited'})) {
                SelectTab $h $main $tab.name
                foreach($format in @(@{item='Text (.txt)…';suffix='txt'},@{item='Markdown (.md)…';suffix='md'},@{item='Subtitles (.srt)…';suffix='srt'},@{item='Data (.json)…';suffix='json'})) {
                    $path=Join-Path $exports ($tab.file+'.'+$format.suffix)
                    $opening=[OwnWindow]::InvokeAsync((FindControl $main 'Export' $button))
                    WaitUntil { [OwnWindow]::FindClass($p.Id,'QMenu',$null) -ne [IntPtr]::Zero } 'Export menu did not open'
                    $menu=[OwnWindow]::FindClass($p.Id,'QMenu',$null)
                    ClickElement $menu (FindControl ([System.Windows.Automation.AutomationElement]::FromHandle($menu)) $format.item ([System.Windows.Automation.ControlType]::MenuItem))
                    FilePicker $p.Id 'Export transcript' $path
                    WaitUntil { (Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -gt 0 } 'Native export not written'
                    if(-not $opening.Wait(5000)) { throw 'Export invocation did not return' }
                }
            }
        }
        # The reopened window comes back on the lecture and version open at close (Edited).
        WaitUntil { (ValueOf $transcript).Current.Value.Contains('Synthetic native correction two') } 'Edited version did not load'
        [OwnWindow]::PostMessage($h,0x10,[IntPtr]::Zero,[IntPtr]::Zero) | Out-Null
        if(-not $p.WaitForExit(10000)) { throw 'Normal GUI close timed out' }
        if($p.ExitCode -ne 0 -or (Test-Path -LiteralPath (Join-Path $Data 'desktop.lock'))) { throw 'GUI close failed or left the library locked' }
        $runs += [ordered]@{exit_code=$p.ExitCode; lock_removed=$true; history_revisions=$entries.Count; newest_revision_preview_matches=$true; history_read_only=$true; older_revision_preview_matches=$true; edited_layer_matches=$true}
    } finally {
        if(-not $p.HasExited) {
            if($h -ne [IntPtr]::Zero) { [OwnWindow]::PostMessage($h,0x10,[IntPtr]::Zero,[IntPtr]::Zero) | Out-Null }
            if(-not $p.WaitForExit(2000)) { $p.Kill(); $p.WaitForExit() }
        }
        $p.Dispose()
    }
}
[ordered]@{runs=$runs; playback=$playback; edits_saved=2; edits_typed_in_transcript=$true; search_selection_matches=$true; reopened_place_and_history_persisted=$true; native_exports=8; dictation_controls=$true; fresh_dictation_defaults='Off / prose (no cleanup model)'; microphone_recording_started=$false; dropdown_selection=$true; dropdowns=@('Speech device (dictation)','Shortcut mode (dictation)')} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Report -Encoding utf8
Write-Output 'Frozen native playback, line seek, typed edits, search, history, export menu and close/reopen passed'
