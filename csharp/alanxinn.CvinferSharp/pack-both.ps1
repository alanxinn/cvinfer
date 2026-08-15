param(
    [string]$Configuration = "Release",
    [string]$Version = "0.1.4"
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$trtProject = Join-Path $projectRoot "alanxinn.CvinferSharp.csproj"
$denglinProject = Join-Path $projectRoot "alanxinn.CvinferSharp.Denglin.csproj"

dotnet pack $trtProject -c $Configuration -p:PackageVersion=$Version `
    -p:BaseOutputPath="$projectRoot\bin\TensorRT\" `
    -p:BaseIntermediateOutputPath="$projectRoot\obj\TensorRT\" `
    -p:MSBuildProjectExtensionsPath="$projectRoot\obj\TensorRT\"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

dotnet pack $denglinProject -c $Configuration -p:PackageVersion=$Version `
    -p:BaseOutputPath="$projectRoot\bin\Denglin\" `
    -p:BaseIntermediateOutputPath="$projectRoot\obj\Denglin\" `
    -p:MSBuildProjectExtensionsPath="$projectRoot\obj\Denglin\"
exit $LASTEXITCODE
