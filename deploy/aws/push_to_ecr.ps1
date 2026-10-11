# Push the locally built radfrac:latest image to Amazon ECR.
#
# Prerequisites: AWS CLI v2 installed and `aws configure` done with your own
# credentials (IAM user/role allowed to use ECR).
#
#   .\deploy\aws\push_to_ecr.ps1 -Region ap-south-1
param(
    [string]$Region = "ap-south-1",
    [string]$Repo = "radfrac"
)
$ErrorActionPreference = "Stop"

$account = (aws sts get-caller-identity --query Account --output text)
$registry = "$account.dkr.ecr.$Region.amazonaws.com"

# Create the repository once (ignore "already exists").
aws ecr describe-repositories --repository-names $Repo --region $Region 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --repository-name $Repo --region $Region | Out-Null
}

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry
docker tag "${Repo}:latest" "$registry/${Repo}:latest"
docker push "$registry/${Repo}:latest"

Write-Host "Pushed: $registry/${Repo}:latest"
