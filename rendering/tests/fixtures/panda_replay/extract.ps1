param([Parameter(Mandatory=$true)][string]$Archive, [Parameter(Mandatory=$true)][string]$ModelDirectory)
$ErrorActionPreference='Stop'
# Text extraction only. GetRawText preserves the exact numeric tokens of each record.
$document=[System.Text.Json.JsonDocument]::Parse([IO.File]::ReadAllText($Archive))
$root=$document.RootElement
$sha=(Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if($sha -ne 'd74e42dd2d3f7735e197b0a845e602217b243a8513c73b7ab09c02995ccb666b'){throw 'foreign archive'}
$indices=@(479,2159)
$records=@(); $digests=@()
foreach($index in $indices){
  $raw=$root.GetProperty('trace')[$index].GetRawText()
  $hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($raw))).ToLowerInvariant()
  $records+=('{"source_index":'+$index+',"record":'+$raw+'}')
  $digests+=@{source_index=$index;raw_record_sha256=$hash}
}
$excerpt='{"schema":"phymeo-panda-archive-excerpt-v1","inputs":'+$root.GetProperty('inputs').GetRawText()+',"records":['+($records -join ',')+']}'
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'trace.json'),$excerpt,[Text.UTF8Encoding]::new($false))
$manifest=[ordered]@{
 schema='phymeo-panda-archive-manifest-v1'; archive_sha256=$sha
 archive_relative_path='artifacts/p65-panda-rotated-offset-20261104-01/phy-results/test_rotated_offset_box_window0/rotated-grip.json'
 archive_record_count=$root.GetProperty('trace').GetArrayLength()
 excerpt_sha256=(Get-FileHash (Join-Path $PSScriptRoot 'trace.json')).Hash.ToLowerInvariant()
 model_sha256=$root.GetProperty('inputs').GetProperty('urdf_sha256').GetString()
 records=$digests
 base_pose=@{position_m=@(0.0,0.0,0.0);orientation_xyzw=@(0.0,0.0,0.0,1.0)}
 base_pose_source='fixed-base load_panda() defaults in archived experiment; not an archived live state identity'
 live_physics_identity=$null
 state_phase='post-step q/cube/xyzw/t; forces and pad_points_before_m belong to previous step input geometry'
}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'manifest.json'),($manifest|ConvertTo-Json -Depth 8)+"`n",[Text.UTF8Encoding]::new($false))
foreach($file in @('panda.urdf','LICENSE','NOTICE','SOURCES.md')){
 Copy-Item -LiteralPath (Join-Path $ModelDirectory $file) -Destination (Join-Path $PSScriptRoot $file)
}
$document.Dispose()