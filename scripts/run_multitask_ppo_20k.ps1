param(
    [int[]]$AgentCounts = @(4),
    [string]$ScalingMode = "fixed_map",
    [string]$ObsVariant = "multi_channel_field+task_id",
    [int]$TotalUpdates = 20000,
    [int]$TargetEpisodes = 0,
    [int]$EvalEpisodes = 8,
    [int]$RecordEvalEpisodes = 1,
    [ValidateSet("gif", "mp4")][string]$RecordFormat = "gif",
    [int]$RecordFps = 8,
    [int]$RecordInterval = 4,
    [int]$ConsoleLogInterval = 25,
    [switch]$NoTensorboard,
    [switch]$ShowEval,
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$taskArgs = @("goal_nav", "coverage", "formation", "risk_nav")
$agentArgs = $AgentCounts | ForEach-Object { $_.ToString() }

$command = @(
    "scripts\train_ppo.py",
    "--config", "configs\policy\ppo_cnn_deepsets_multitask_20k.yaml",
    "--env-config", "configs\env\multitask.yaml",
    "--tasks"
) + $taskArgs + @(
    "--agent_counts"
) + $agentArgs + @(
    "--scaling_mode", $ScalingMode,
    "--obs_variant", $ObsVariant,
    "--total_updates", $TotalUpdates.ToString(),
    "--target_episodes", $TargetEpisodes.ToString(),
    "--eval_episodes", $EvalEpisodes.ToString(),
    "--console_log_interval", $ConsoleLogInterval.ToString(),
    "--record_eval_episodes", $RecordEvalEpisodes.ToString(),
    "--record_format", $RecordFormat,
    "--record_fps", $RecordFps.ToString(),
    "--record_interval", $RecordInterval.ToString()
)

if ($NoTensorboard) {
    $command += "--no-tensorboard"
}
else {
    $command += "--tensorboard"
}

if ($ShowEval) {
    $command += "--no-headless"
}
else {
    $command += "--headless"
}

Write-Host "Running:" $PythonExe ($command -join " ")
& $PythonExe @command
