# arcinfer launcher: a window with every model (and variant) server.exe finds, what each supports, how its automatic
# context was counted (GPU memory by part), every server option, and the running server (state, GPU memory,
# requests and speeds). Start / Stop go through arc-server.ps1, so the server keeps running after the window closes.
# Start it with arc-launcher.cmd (project folder) or:  pwsh -STA -File scripts\arc-launcher.ps1
# (ASCII only: Windows PowerShell 5.1 reads scripts without a BOM as ANSI)
param([int]$Port = 8090)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$root = Split-Path $PSScriptRoot -Parent
$arcServer = Join-Path $PSScriptRoot "arc-server.ps1"
$stateDir = Join-Path $env:LOCALAPPDATA "arcinfer"
New-Item -ItemType Directory -Force $stateDir | Out-Null
# the engine (arc-server.ps1 -Zero): arcinfer (bin\server.exe) or arcinferzero
# (bin\serverz.exe: MoE experts streamed from the SSD); the model list and its contexts come from that executable
$script:engine = "zero"  # (the default engine since 2026-10-16; arcinfer = OpenCL)
function Exe { Join-Path $root $(if ($script:engine -eq "zero") { "bin\serverz.exe" } else { "bin\server.exe" }) }
$exe = Exe
$stateFile = Join-Path $stateDir "server.json"
$serverLog = Join-Path $stateDir "server.log"
$shell = (Get-Process -Id $PID).Path  # the PowerShell running this window (pwsh or powershell)
function Base { "http://127.0.0.1:$([int]$portBox.Value)" }
function GiB($b) { if ($null -eq $b) { return "-" }; "{0:N2} GiB" -f ([double]$b / 1GB) }

# background work as processes writing files, collected by a timer: the model list (server.exe --list-models; up to
# a minute per model the first time, while the automatic context is computed) and arc-server.ps1 start / stop
$script:listProc = $null
$script:listOut = Join-Path $env:TEMP "arc-launcher-models.json"
$script:actProc = $null
$script:actOut = Join-Path $env:TEMP "arc-launcher-action.log"
$script:actErr = Join-Path $env:TEMP "arc-launcher-action.err"
$script:models = @()
$script:running = $null
$script:props = $null

$form = New-Object System.Windows.Forms.Form
$form.Text = "arcinfer launcher"
$form.Size = New-Object System.Drawing.Size(1320, 860)
$form.MinimumSize = New-Object System.Drawing.Size(1000, 640)
$form.StartPosition = "CenterScreen"
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

function New-Label($t, $bold = $false) {
  $l = New-Object System.Windows.Forms.Label; $l.Text = $t; $l.AutoSize = $true
  $l.Margin = New-Object System.Windows.Forms.Padding(3, 7, 3, 0)
  if ($bold) { $l.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold) }
  $l
}
function New-Button($t, $w = 110) {
  $b = New-Object System.Windows.Forms.Button; $b.Text = $t; $b.Width = $w; $b.Height = 30
  $b.Margin = New-Object System.Windows.Forms.Padding(4, 3, 4, 3); $b
}
function New-Num($min, $max, $val, $dec = 0, $w = 80) {
  $n = New-Object System.Windows.Forms.NumericUpDown; $n.Minimum = $min; $n.Maximum = $max; $n.DecimalPlaces = $dec
  $n.Value = $val; $n.Width = $w; $n.Margin = New-Object System.Windows.Forms.Padding(3, 3, 3, 3); $n
}
function New-Text($w = 80) { $t = New-Object System.Windows.Forms.TextBox; $t.Width = $w; $t.Margin = New-Object System.Windows.Forms.Padding(3, 3, 3, 3); $t }
function New-Combo($items, $w = 90) {
  $c = New-Object System.Windows.Forms.ComboBox; $c.DropDownStyle = "DropDownList"; $c.Width = $w
  $c.Margin = New-Object System.Windows.Forms.Padding(3, 3, 3, 3); [void]$c.Items.AddRange($items); $c.SelectedIndex = 0; $c
}
function New-Check($t) { $c = New-Object System.Windows.Forms.CheckBox; $c.Text = $t; $c.AutoSize = $true; $c.Margin = New-Object System.Windows.Forms.Padding(3, 5, 3, 0); $c }
function New-Info {
  $t = New-Object System.Windows.Forms.TextBox; $t.Multiline = $true; $t.ReadOnly = $true; $t.Dock = "Fill"
  $t.ScrollBars = "Vertical"; $t.BackColor = [System.Drawing.SystemColors]::Window
  $t.Font = New-Object System.Drawing.Font("Consolas", 9); $t
}

# ---- status line
$status = New-Object System.Windows.Forms.Label
$status.Dock = "Top"; $status.Height = 40
$status.Padding = New-Object System.Windows.Forms.Padding(10, 10, 10, 4)
$status.Font = New-Object System.Drawing.Font("Segoe UI", 10.5, [System.Drawing.FontStyle]::Bold)
$status.Text = "Checking the server..."

# ---- model table
$grid = New-Object System.Windows.Forms.DataGridView
$grid.Dock = "Fill"; $grid.ReadOnly = $true; $grid.AllowUserToAddRows = $false; $grid.AllowUserToDeleteRows = $false
$grid.AllowUserToResizeRows = $false; $grid.SelectionMode = "FullRowSelect"; $grid.MultiSelect = $false
$grid.RowHeadersVisible = $false; $grid.AutoSizeColumnsMode = "Fill"; $grid.BackgroundColor = [System.Drawing.SystemColors]::Window
foreach ($c in @(
    @("model", "Model", 30), @("variant", "Variant", 11), @("mtp", "MTP", 5), @("vision", "Images", 6),
    @("ctx", "Auto context", 10), @("vram", "VRAM (full context)", 11), @("size", "File", 7),
    @("id", "Id (Pi /model)  [shortcut]", 20), @("compat", "Supported", 11))) {
  $col = New-Object System.Windows.Forms.DataGridViewTextBoxColumn
  $col.Name = $c[0]; $col.HeaderText = $c[1]; $col.FillWeight = $c[2]
  [void]$grid.Columns.Add($col)
}

# ---- options (every server option arc-server.ps1 passes on). Each field shows what the server picks for the
# selected model; a field you change turns yellow and is passed on (the others are left to the server)
$portBox = New-Num 1024 65535 $Port 0 70
$ctxBox = New-Text 80
$chunkBox = New-Combo @("512", "1024", "2048", "4096")
$draftBox = New-Combo @("1", "2", "3") 60
$thinkBox = New-Num 0 262144 8192 0 80
$imgMinBox = New-Num 0 16384 0 0 70
$imgMaxBox = New-Num 0 16384 0 0 70
$budgetBox = New-Num 0 64 0 1 70
$sessBox = New-Num 0 512 0 0 70
$sessDirBox = New-Text 170
$prefixBox = New-Num 0 512 8 0 70
$noPrefixChk = New-Check "no prefix cache"
$noSwapChk = New-Check "keep the loaded model (--no-swap: requests do not switch models)"
$netChk = New-Check "reachable from other computers (0.0.0.0)"
$f32Chk = New-Check "f32 attention (tests)"
$engineBox = New-Combo @("arcinfer", "arcinferzero") 120
$engineBox.SelectedIndex = $(if ($script:engine -eq "zero") { 1 } else { 0 })

$opt = New-Object System.Windows.Forms.TableLayoutPanel
$opt.Dock = "Fill"; $opt.ColumnCount = 2; $opt.AutoScroll = $true
[void]$opt.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("AutoSize")))
[void]$opt.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
function Add-Opt($label, $ctl, $hint = "", [switch]$Advanced) {
  $p = New-Object System.Windows.Forms.FlowLayoutPanel; $p.AutoSize = $true; $p.WrapContents = $false
  $p.Margin = New-Object System.Windows.Forms.Padding(0)
  [void]$p.Controls.Add($ctl)
  if ($hint) { $h = New-Label $hint; $h.ForeColor = [System.Drawing.Color]::DimGray; [void]$p.Controls.Add($h) }
  $l = New-Label $label
  $opt.Controls.Add($l); $opt.Controls.Add($p)
  if ($Advanced) { $script:advRows += , @($l, $p) }
}
$script:advRows = @()
$optNote = New-Label "Filled in with the best values for the selected model. Change one to override it (yellow)."
$optNote.ForeColor = [System.Drawing.Color]::DimGray
$opt.Controls.Add($optNote); $opt.SetColumnSpan($optNote, 2)
# the options most people change; the rest behind "Show advanced options"
Add-Opt "Engine" $engineBox "arcinferzero: Level Zero; MoE models (35B) stream experts from the SSD: 262K with images + MTP"
Add-Opt "Context (tokens)" $ctxBox "auto: the largest that fits"
# the addresses other apps connect to (OpenAI / Anthropic / Ollama formats), with a Copy button each
$apiPanel = New-Object System.Windows.Forms.FlowLayoutPanel
$apiPanel.AutoSize = $true; $apiPanel.WrapContents = $false; $apiPanel.Margin = New-Object System.Windows.Forms.Padding(0)
$apiBox = New-Text 230; $apiBox.ReadOnly = $true
$apiFmt = New-Combo @("OpenAI (/v1)", "Anthropic (/v1/messages)", "Ollama") 170
$apiFmt.SelectedIndex = 0
$apiCopy = New-Button "Copy" 60
function Update-Api {
  $b = Base
  $apiBox.Text = switch ($apiFmt.SelectedIndex) { 1 { "$b/v1/messages" } 2 { $b } default { "$b/v1" } }
}
$apiFmt.Add_SelectedIndexChanged({ Update-Api })
$portBox.Add_ValueChanged({ Update-Api })
$apiCopy.Add_Click({ Update-Api; [System.Windows.Forms.Clipboard]::SetText($apiBox.Text) })
[void]$apiPanel.Controls.Add($apiFmt); [void]$apiPanel.Controls.Add($apiBox); [void]$apiPanel.Controls.Add($apiCopy)
Update-Api
Add-Opt "API address" $apiPanel "for other apps; API key: anything; model: the id in the list"
Add-Opt "MTP drafts per step" $draftBox
Add-Opt "Thinking budget (tokens)" $thinkBox
Add-Opt "Prompt cache (GB on disk)" $sessBox "whole conversations, kept when switching sessions; 0 = off"
Add-Opt "" $netChk
$advChk = New-Check "Show advanced options"
$advChk.Margin = New-Object System.Windows.Forms.Padding(3, 10, 3, 4)
$opt.Controls.Add($advChk); $opt.SetColumnSpan($advChk, 2)
Add-Opt "Port" $portBox -Advanced
Add-Opt "Prompt chunk" $chunkBox "smaller = more context, slower prompts" -Advanced
Add-Opt "Image tokens min / max" $imgMinBox -Advanced
Add-Opt "" $imgMaxBox -Advanced
Add-Opt "Resident W8 weights (GB)" $budgetBox "0 = chosen with the context" -Advanced
Add-Opt "Prompt cache folder" $sessDirBox "empty = %LOCALAPPDATA%\arcinfer\sessions" -Advanced
Add-Opt "Prefix cache (GB on disk)" $prefixBox "system prompts per project" -Advanced
Add-Opt "" $noPrefixChk -Advanced
Add-Opt "" $noSwapChk -Advanced
Add-Opt "" $f32Chk -Advanced
function Show-Advanced([bool]$on) { foreach ($r in $script:advRows) { $r[0].Visible = $on; $r[1].Visible = $on } }
$advChk.Add_CheckedChanged({ Show-Advanced $advChk.Checked })
Show-Advanced $false

# overrides: field name -> value the user set (passed on by Start, saved by Set as default)
$script:over = @{}
$script:filling = $false
$fields = [ordered]@{
  port = $portBox; ctx = $ctxBox; chunk = $chunkBox; drafts = $draftBox; think = $thinkBox; img_min = $imgMinBox
  img_max = $imgMaxBox; budget = $budgetBox; prompt_cache = $sessBox; prompt_cache_dir = $sessDirBox
  prefix_cache = $prefixBox; no_prefix = $noPrefixChk; no_swap = $noSwapChk; network = $netChk; attn_f32 = $f32Chk
}
function Get-Field($k) {
  $c = $fields[$k]
  switch ($c.GetType().Name) {
    "TextBox" { return $c.Text.Trim() }
    "NumericUpDown" { return [double]$c.Value }
    "ComboBox" { return [string]$c.SelectedItem }
    default { return [bool]$c.Checked }
  }
}
function Put-Field($k, $v) {
  $c = $fields[$k]
  $script:filling = $true
  try {
    switch ($c.GetType().Name) {
      "TextBox" { $c.Text = "$v" }
      "NumericUpDown" { $c.Value = [decimal][math]::Max([double]$c.Minimum, [math]::Min([double]$c.Maximum, [double]$v)) }
      "ComboBox" { $i = $c.Items.IndexOf([string]$v); if ($i -ge 0) { $c.SelectedIndex = $i } }
      default { $c.Checked = [bool]$v }
    }
  } finally { $script:filling = $false }
}
function Show-Mark($k) {
  $c = $fields[$k]
  $on = $script:over.ContainsKey($k)
  if ($c -is [System.Windows.Forms.CheckBox]) { $c.ForeColor = if ($on) { [System.Drawing.Color]::DarkGoldenrod } else { [System.Drawing.SystemColors]::ControlText } }
  else { $c.BackColor = if ($on) { [System.Drawing.Color]::LightYellow } else { [System.Drawing.SystemColors]::Window } }
}
$onEdit = {
  param($sender, $e)
  if ($script:filling) { return }
  $k = $sender.Tag
  $script:over[$k] = Get-Field $k
  Show-Mark $k
}
foreach ($k in $fields.Keys) {
  $c = $fields[$k]
  $c.Tag = $k
  switch ($c.GetType().Name) {
    "TextBox" { $c.Add_TextChanged($onEdit) }
    "NumericUpDown" { $c.Add_ValueChanged($onEdit) }
    "ComboBox" { $c.Add_SelectedIndexChanged($onEdit) }
    default { $c.Add_CheckedChanged($onEdit) }
  }
}
# the selected model's best values (what the server picks itself), into every field not overridden
function Fill-Defaults {
  $m = if ($grid.SelectedRows.Count) { $grid.SelectedRows[0].Tag } else { $null }
  $d = $null
  if ($m) { $d = $m.defaults }
  $v = [ordered]@{
    port = $Port; ctx = $(if ($m -and $m.ctx) { "$($m.ctx)" } else { "auto" })
    chunk = $(if ($d) { $d.chunk } else { 4096 }); drafts = $(if ($d -and $d.mtp_drafts) { $d.mtp_drafts } else { 2 })
    think = 8192; img_min = $(if ($d) { $d.image_min } else { 0 }); img_max = $(if ($d) { $d.image_max } else { 0 })
    budget = 0; prompt_cache = 16; prompt_cache_dir = ""; prefix_cache = 8; no_prefix = $false; no_swap = $false
    network = $false; attn_f32 = $false
  }
  foreach ($k in $fields.Keys) {
    Put-Field $k $(if ($script:over.ContainsKey($k)) { $script:over[$k] } else { $v[$k] })
    Show-Mark $k
  }
  $draftBox.Enabled = [bool]($m -and $m.mtp)
  $imgMinBox.Enabled = $imgMaxBox.Enabled = [bool]($m -and $m.vision)
}

$detail = New-Info
$monitor = New-Info

function New-Group($title, $ctl) {
  $g = New-Object System.Windows.Forms.GroupBox; $g.Text = $title; $g.Dock = "Fill"
  $g.Padding = New-Object System.Windows.Forms.Padding(6); [void]$g.Controls.Add($ctl); $g
}
$bottom = New-Object System.Windows.Forms.TableLayoutPanel
$bottom.Dock = "Fill"; $bottom.ColumnCount = 3; $bottom.RowCount = 1
foreach ($w in 36, 32, 32) { [void]$bottom.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", $w))) }
$bottom.Controls.Add((New-Group "Server options (used by Start)" $opt), 0, 0)
$bottom.Controls.Add((New-Group "Selected model: features and how the auto context is computed" $detail), 1, 0)
$bottom.Controls.Add((New-Group "Running server" $monitor), 2, 0)

$split = New-Object System.Windows.Forms.SplitContainer
$split.Dock = "Fill"; $split.Orientation = "Horizontal"
[void]$split.Panel1.Controls.Add($grid)
[void]$split.Panel2.Controls.Add($bottom)

# ---- buttons and message line
$buttons = New-Object System.Windows.Forms.FlowLayoutPanel
$buttons.Dock = "Bottom"; $buttons.Height = 42; $buttons.Padding = New-Object System.Windows.Forms.Padding(6, 4, 6, 2)
$startBtn = New-Button "Start" 130
$startBtn.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
$stopBtn = New-Button "Stop server" 120
$refreshBtn = New-Button "Refresh list" 120
$defBtn = New-Button "Set as default" 120
$resetBtn = New-Button "Reset options" 110
$webBtn = New-Button "Open web UI"
$logBtn = New-Button "Open log"
$buttons.Controls.AddRange(@($startBtn, $stopBtn, $defBtn, $resetBtn, $refreshBtn, $webBtn, $logBtn))
$msg = New-Object System.Windows.Forms.Label
$msg.Dock = "Bottom"; $msg.Height = 24; $msg.Padding = New-Object System.Windows.Forms.Padding(10, 4, 10, 0)
$msg.ForeColor = [System.Drawing.Color]::DimGray
$msg.Text = "Reading the model list..."

$form.Controls.Add($split)
$form.Controls.Add($buttons)
$form.Controls.Add($msg)
$form.Controls.Add($status)
$form.Add_Load({ $split.SplitterDistance = [int]($split.Height * 0.45) })

# ---- the selected model: what it supports and how --ctx auto counted the GPU memory
function Show-Detail {
  if ($grid.SelectedRows.Count -eq 0) { $detail.Text = ""; return }
  $m = $grid.SelectedRows[0].Tag
  $L = New-Object System.Collections.Generic.List[string]
  $L.Add("$($m.name)")
  $L.Add("file    $($m.file)  ({0:N1} GB)" -f ($m.bytes / 1e9))
  $L.Add("arch    $($m.arch)  -  " + $(if ($m.supported) { "supported by arcinfer" } else { "NOT supported by arcinfer" }))
  $L.Add("id      $($m.alias)" + $(if ($m.short) { "   (shortcut: pi-arc $($m.short))" } else { "" }))
  $L.Add("variant $(if ($m.tags) { $m.tags } else { '-' })")
  $L.Add("MTP / drafter: $(if ($m.mtp) { 'yes (drafts checked by the main model: same output, faster)' } else { 'no' })")
  $L.Add("images: $(if ($m.vision) { 'yes (encoder from the mmproj file)' } else { 'no' })")
  $L.Add("max context length (model): {0:N0} tokens" -f $m.ctx_train)
  $mm = $m.memory
  if ($m.supported -and $mm) {
    $budget = $mm.device - $mm.keep_free - $mm.snapshots - $mm.encoders
    $L.Add("")
    $L.Add("Auto context (--ctx auto):")
    $L.Add("  GPU memory (Intel setting)    {0,12}" -f (GiB $mm.device))
    $L.Add("  - kept for Windows / apps     {0,12}" -f (GiB $mm.keep_free))
    $L.Add("  - conversation snapshots      {0,12}" -f (GiB $mm.snapshots))
    $L.Add("  - image / audio encoders      {0,12}" -f (GiB $mm.encoders))
    $L.Add("  = for the model               {0,12}" -f (GiB $budget))
    $L.Add("  - weights{0}{1,12}" -f $(if ($mm.resident) { " (+ W8 copies)     " } else { "                   " }), (GiB $mm.weights))
    $L.Add("  - MTP block / drafter         {0,12}" -f (GiB $mm.mtp))
    $L.Add("  - working buffers             {0,12}" -f (GiB $mm.buffers))
    $L.Add("  = left for the context        {0,12}" -f (GiB ($budget - $mm.weights - $mm.mtp - $mm.buffers)))
    $L.Add("  context per 1,000 tokens      {0,12}" -f ("{0:N1} MiB" -f ($mm.context_per_1k / 1MB)))
    $L.Add("  -> {0:N0} tokens (KV {1}; multiples of 4,096, at most the max context length)" -f $m.ctx, (GiB $mm.context))
  } elseif ($m.supported) {
    $L.Add(""); $L.Add("(no memory details yet: Refresh list)")
  }
  $detail.Text = $L -join "`r`n"
}

# ---- the running server: /v1/models, /props (memory, stats)
function Update-Status {
  $id = $null; $p = $null; $busyAct = $script:actProc -and -not $script:actProc.HasExited
  try {
    $d = @((Invoke-RestMethod "$(Base)/v1/models" -TimeoutSec 1).data)
    $l = @($d | Where-Object { $_.loaded -eq $true })
    $id = if ($l.Count) { $l[0].id } else { $d[0].id }
    try { $p = Invoke-RestMethod "$(Base)/props" -TimeoutSec 1 } catch {}
  } catch {}
  $script:running = $id; $script:props = $p
  $st = $null
  if (Test-Path $stateFile) { try { $st = Get-Content $stateFile -Raw | ConvertFrom-Json } catch {} }
  if ($id) {
    $up = ""
    if ($st -and $st.started) { try { $up = "  |  up {0:hh\:mm\:ss}" -f ((Get-Date) - [datetime]$st.started) } catch {} }
    $busy = if ($p -and $p.stats.busy) { "  |  ANSWERING" } else { "" }
    $status.Text = ("Server ON  ({0})  -  {1}  |  context {2:N0}{3}{4}" -f (Base), $id, $(if ($p) { $p.n_ctx } else { 0 }), $up, $busy)
    $status.ForeColor = [System.Drawing.Color]::DarkGreen
  } elseif ($busyAct) {
    $status.Text = "Server loading / stopping..."
    $status.ForeColor = [System.Drawing.Color]::DarkOrange
  } else {
    $status.Text = "Server OFF"
    $status.ForeColor = [System.Drawing.Color]::DarkRed
  }
  $stopBtn.Enabled = [bool]$id -and -not $busyAct
  $startBtn.Enabled = -not $busyAct -and $grid.SelectedRows.Count -gt 0
  # monitoring text
  if ($id -and $p) {
    $L = New-Object System.Collections.Generic.List[string]
    $mem = $p.memory
    $L.Add("model    $id")
    $L.Add("context  {0:N0} tokens, {1:N0} filled   chunk {2}" -f $p.n_ctx, $p.stats.position, $p.chunk)
    $L.Add("MTP      {0}   images {1}   audio {2}" -f $(if ($p.mtp) { "yes ($($p.mtp_drafts) drafts)" } else { "no" }), $(if ($p.vision) { "yes" } else { "no" }), $(if ($p.audio) { "yes" } else { "no" }))
    if ($mem) {
      $L.Add("")
      $L.Add("VRAM used by the server  {0,12}  of {1}" -f (GiB $mem.total), (GiB $mem.device))
      $L.Add("  weights                {0,12}" -f (GiB $mem.weights))
      $L.Add("  context (KV)           {0,12}" -f (GiB $mem.context))
      $L.Add("  MTP / drafter          {0,12}" -f (GiB $mem.mtp))
      $L.Add("  working buffers        {0,12}" -f (GiB $mem.buffers))
      $L.Add("  encoders (loaded)      {0,12}" -f (GiB $mem.encoders))
      $pct = [math]::Min(100, [math]::Round(100 * $mem.total / $mem.device))
      $L.Add(("  [" + ("#" * [int]($pct / 4)) + ("." * (25 - [int]($pct / 4))) + "] {0}%") -f $pct)
    }
    $L.Add("")
    $L.Add("requests done  $($p.stats.requests)" + $(if ($p.stats.busy) { "   (one in progress)" } else { "" }))
    $t = $p.stats.last
    if ($t) {
      $L.Add("last request:")
      $L.Add("  prompt  {0} new tokens ({1} from cache), {2:N1} tok/s" -f $t.prompt_n, $t.cache_n, $t.prompt_per_second)
      $L.Add("  answer  {0} tokens, {1:N1} tok/s  ({2})" -f $t.predicted_n, $t.predicted_per_second, $t.finish)
      if ($t.draft_n) { $L.Add("  MTP     {0}/{1} drafts accepted ({2:N0}%)" -f $t.draft_n_accepted, $t.draft_n, (100 * $t.draft_n_accepted / [math]::Max(1, $t.draft_n))) }
    }
    if ($st) { $L.Add(""); $L.Add("started with: ctx $($st.ctx), chunk $($st.chunk), $(if ($st.extra) { $st.extra } else { 'default options' })") }
    $monitor.Text = $L -join "`r`n"
  } elseif (-not $id) {
    $monitor.Text = "The server is not running."
  }
  foreach ($r in $grid.Rows) {
    $m = $r.Tag
    $on = $id -and $m -and $m.alias -eq $id
    $r.DefaultCellStyle.BackColor = if ($on) { [System.Drawing.Color]::FromArgb(214, 240, 214) } elseif ($m -and -not $m.supported) { [System.Drawing.Color]::FromArgb(238, 238, 238) } else { [System.Drawing.SystemColors]::Window }
  }
}

# ---- model list
function Start-List {
  if ($script:listProc -and -not $script:listProc.HasExited) { return }
  $script:exe = Exe
  if (-not (Test-Path $script:exe)) { $msg.Text = "$(Split-Path $script:exe -Leaf) not found: build it first (.\build.ps1 $(if ($script:engine -eq 'zero') { 'zero\serverz.cpp' } else { 'tools\server.cpp' }))"; return }
  $msg.Text = "Reading the model list (up to a minute per model the first time: the auto context is computed)..."
  Remove-Item $script:listOut -ErrorAction SilentlyContinue
  $script:listProc = Start-Process -FilePath $script:exe -ArgumentList "--list-models" -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $script:listOut -RedirectStandardError (Join-Path $env:TEMP "arc-launcher-models.err")
}
function Show-List {
  $sel = if ($grid.SelectedRows.Count) { $grid.SelectedRows[0].Tag.alias } else { $script:running }
  $selPath = if (-not $sel -and (Test-Path $lastFile)) { (Get-Content $lastFile -Raw).Trim() } else { $null }
  $grid.Rows.Clear()
  foreach ($m in $script:models) {
    $mm = $m.memory
    $vram = if ($m.supported -and $mm) { GiB ($mm.weights + $mm.mtp + $mm.buffers + $mm.context + $mm.snapshots + $mm.encoders) } else { "-" }
    $i = $grid.Rows.Add($m.name + "  -  " + $m.file, $(if ($m.tags) { $m.tags } else { "-" }),
      $(if ($m.mtp) { "yes" } else { "-" }), $(if ($m.vision) { "yes" } else { "-" }),
      $(if ($m.supported -and $m.ctx) { "{0:N0}" -f $m.ctx } else { "-" }), $vram, ("{0:N1} GB" -f ($m.bytes / 1e9)),
      $(if ($m.short) { "$($m.alias)  [$($m.short)]" } else { $m.alias }),
      $(if ($m.supported) { "yes" } else { "no ($($m.arch))" }))
    $grid.Rows[$i].Tag = $m
    if (($sel -and $m.alias -eq $sel) -or ($selPath -and -not $sel -and $m.path -and [IO.Path]::GetFullPath($m.path) -eq [IO.Path]::GetFullPath($selPath))) {
      $grid.Rows[$i].Selected = $true; $grid.FirstDisplayedScrollingRowIndex = $i; if (-not $sel) { $sel = $m.alias }
    }
  }
  $n = @($script:models | Where-Object { $_.supported }).Count
  $msg.Text = "$n models / variants can run. Select a row and Start (or double-click it). The server keeps running when this window is closed."
  Show-Detail
  Fill-Defaults
  Update-Status
}

# ---- start / stop through arc-server.ps1 (a separate process: the server outlives this window)
function Start-Action([string[]]$argList) {
  if ($script:actProc -and -not $script:actProc.HasExited) { return }
  Remove-Item $script:actOut, $script:actErr -ErrorAction SilentlyContinue
  $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$arcServer`"") + $argList + @("-Port", [int]$portBox.Value)
  $script:actProc = Start-Process -FilePath $shell -ArgumentList $a -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $script:actOut -RedirectStandardError $script:actErr
  Update-Status
}
function Start-Selected {
  if ($grid.SelectedRows.Count -eq 0) { return }
  $m = $grid.SelectedRows[0].Tag
  if (-not $m.supported) {
    [void][System.Windows.Forms.MessageBox]::Show("arcinfer does not support this model yet (architecture $($m.arch)).", "arcinfer"); return
  }
  $a = @("start", "-Model", $m.alias)
  if ($script:engine -eq "zero") { $a += "-Zero" } else { $a += "-OpenCL" }
  foreach ($k in @($script:over.Keys)) {
    $v = $script:over[$k]
    switch ($k) {
      "ctx" {
        if ($v -and $v -ne "auto") {
          if ($v -notmatch '^\d+$') { [void][System.Windows.Forms.MessageBox]::Show("Context: a number of tokens, or auto.", "arcinfer"); return }
          $a += @("-Ctx", $v)
        } elseif ($v -eq "auto") { $a += @("-Ctx", "auto") }
      }
      "chunk" { $a += @("-Chunk", $v) }
      "drafts" { if ($m.mtp) { $a += @("-MtpDrafts", $v) } }
      "think" { $a += @("-ThinkBudget", [int]$v) }
      "img_min" { if ($m.vision) { $a += @("-ImageMinTokens", [int]$v) } }
      "img_max" { if ($m.vision) { $a += @("-ImageMaxTokens", [int]$v) } }
      "budget" { if ($v -gt 0) { $a += @("-BudgetGb", [string]$v) } }
      "prompt_cache" { $a += @("-SessionCacheGb", [int]$v) }  # (0 = off)
      "prompt_cache_dir" { if ($v) { $a += @("-SessionDir", "`"$v`"") } }
      "prefix_cache" { $a += @("-PrefixCacheGb", [int]$v) }
      "no_prefix" { if ($v) { $a += "-NoPrefixCache" } }
      "no_swap" { if ($v) { $a += "-NoSwap" } }
      "network" { if ($v) { $a += @("-ListenHost", "0.0.0.0") } }
      "attn_f32" { if ($v) { $a += "-AttnF32" } }
    }
  }
  $msg.Text = "Starting $($m.alias)... (loading takes 5-40 s; 35B up to 1-3 min)"
  Start-Action $a
}

# ---- defaults: Set as default keeps the overridden fields (launcher.json, applied whenever the window opens) and
# makes the selected model the default one (last-model.txt: also pi-arc's choice when no model is given)
$settingsFile = Join-Path $stateDir "launcher.json"
$lastFile = Join-Path $stateDir "last-model.txt"
if (Test-Path $settingsFile) {
  try {
    $o = Get-Content $settingsFile -Raw | ConvertFrom-Json
    foreach ($p in $o.PSObject.Properties) { if ($fields.Contains($p.Name)) { $script:over[$p.Name] = $p.Value } }
  } catch {}
}
if (@($script:over.Keys | Where-Object { $_ -notin "ctx", "drafts", "think", "prompt_cache", "network" }).Count) { $advChk.Checked = $true }
$defBtn.Add_Click({
    $script:over | ConvertTo-Json | Set-Content $settingsFile -Encoding utf8
    $n = $script:over.Count
    $t = $(if ($n) { "$n changed option(s) saved as default." } else { "No changed options: every model uses its best values." })
    if ($grid.SelectedRows.Count -gt 0 -and $grid.SelectedRows[0].Tag.supported) {
      $m = $grid.SelectedRows[0].Tag
      Set-Content $lastFile $m.path -Encoding utf8
      $t = "Default model: $($m.alias) (also for pi-arc without a model). " + $t
    }
    $msg.Text = $t
  })
$resetBtn.Add_Click({
    $script:over = @{}
    Fill-Defaults
    $msg.Text = "Options back to the best values for each model (Set as default to keep it that way)."
  })

$startBtn.Add_Click({ Start-Selected })
$grid.Add_CellDoubleClick({ param($s, $e) if ($e.RowIndex -ge 0) { Start-Selected } })
$engineBox.Add_SelectedIndexChanged({
  $script:engine = if ($engineBox.SelectedIndex -eq 1) { "zero" } else { "arcinfer" }
  Start-List
})
$grid.Add_SelectionChanged({ Show-Detail; Fill-Defaults; $startBtn.Enabled = -not ($script:actProc -and -not $script:actProc.HasExited) -and $grid.SelectedRows.Count -gt 0 })
$stopBtn.Add_Click({
    if ([System.Windows.Forms.MessageBox]::Show("Stop the server ($($script:running))? Its GPU memory is freed.", "arcinfer", "YesNo") -eq "Yes") {
      $msg.Text = "Stopping the server..."
      Start-Action @("stop")
    }
  })
$refreshBtn.Add_Click({ Start-List })
$webBtn.Add_Click({ Start-Process "$(Base)/" })
$logBtn.Add_Click({ if (Test-Path $serverLog) { Start-Process notepad.exe -ArgumentList "`"$serverLog`"" } })

# ---- timer: collect background results; the server state every 2 s
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 500
$script:tick = 0
$timer.Add_Tick({
    if ($script:listProc -and $script:listProc.HasExited) {
      $script:listProc = $null
      try {
        $script:models = @(Get-Content $script:listOut -Raw | ConvertFrom-Json | ForEach-Object { $_ })
        Show-List
      } catch { $msg.Text = "Could not read the model list: $_" }
    }
    if ($script:actProc) {
      $last = if (Test-Path $script:actOut) { Get-Content $script:actOut -Tail 1 -ErrorAction SilentlyContinue } else { $null }
      if ($last) { $msg.Text = "$last" }
      if ($script:actProc.HasExited) {
        if ($script:actProc.ExitCode -ne 0) {
          $err = Get-Content $script:actErr -Raw -ErrorAction SilentlyContinue
          $tail = Get-Content $script:actOut -Tail 8 -ErrorAction SilentlyContinue
          [void][System.Windows.Forms.MessageBox]::Show("Failed:`n$(($tail -join "`n"))`n$err", "arcinfer")
        }
        $script:actProc = $null
        Update-Status
      }
    }
    if (($script:tick++ % 4) -eq 0) { Update-Status }
  })

$form.Add_Shown({ Update-Status; Start-List; $timer.Start() })
$form.Add_FormClosing({ $timer.Stop() })  # (a running server and a start in progress keep going)
[void]$form.ShowDialog()
