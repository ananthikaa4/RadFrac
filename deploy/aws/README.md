# Deploying RadFrac on AWS (EC2 + ECR)

The container serves the **Streamlit UI** (port 8501) and the **FastAPI API**
(port 8000) from one image. On EC2 we map the UI to port 80 so the public URL is
simply `http://<public-ip>/`.

```
 laptop ──docker push──► Amazon ECR ──docker pull──► EC2 (Docker)
                                                     ├─ :80   → Streamlit UI
                                                     └─ :8000 → FastAPI (/docs)
```

## 0. Prerequisites (one time, on your laptop)

1. An AWS account. Install the **AWS CLI v2** and run `aws configure`
   (access key of an IAM user that can use ECR and EC2).
2. Docker Desktop running, and the image built:

   ```powershell
   cd radfrac
   docker build -t radfrac:latest .
   docker run --rm -p 8501:8501 -p 8000:8000 radfrac:latest   # local smoke test
   ```

## 1. Push the image to ECR

```powershell
.\deploy\aws\push_to_ecr.ps1 -Region ap-south-1
```

It prints the image URI, e.g. `123456789012.dkr.ecr.ap-south-1.amazonaws.com/radfrac:latest`.

## 2. Launch the EC2 instance (console)

| Setting | Value |
|---|---|
| AMI | Amazon Linux 2023 (x86_64) |
| Instance type | **t3.medium** (2 vCPU / 4 GB) — TensorFlow + 5 models need ~2 GB RAM; t3.micro is too small |
| Storage | 30 GB gp3 (the image is ~5 GB uncompressed) |
| IAM instance profile | a role with **AmazonEC2ContainerRegistryReadOnly** |
| Security group inbound | TCP **80** from 0.0.0.0/0 (UI), TCP **8000** from 0.0.0.0/0 (API), TCP 22 from *your IP only* |
| User data | paste `deploy/aws/ec2_user_data.sh` after editing `REGION` and `IMAGE` |

Wait ~3–4 minutes after launch (Docker install + image pull + model loading),
then open:

* UI → `http://<EC2-public-IPv4>/`
* API docs → `http://<EC2-public-IPv4>:8000/docs`
* Health → `http://<EC2-public-IPv4>:8000/health` (lists `advanced_models`)

Use an **Elastic IP** if you need the address to survive instance stop/start.

## 3. Updating

```powershell
docker build -t radfrac:latest . ; .\deploy\aws\push_to_ecr.ps1
```

then on the instance (EC2 Instance Connect / SSH):

```bash
sudo docker pull <IMAGE> && sudo docker rm -f radfrac && \
sudo docker run -d --name radfrac --restart unless-stopped -p 80:8501 -p 8000:8000 \
  -v radfrac-runtime:/app/runtime <IMAGE>
```

## Alternative without ECR

Copy the project to the instance and build there:

```bash
scp -r radfrac ec2-user@<ip>:~ && ssh ec2-user@<ip>
sudo dnf install -y docker && sudo systemctl enable --now docker
cd radfrac && sudo docker build -t radfrac:latest .
sudo docker run -d --name radfrac --restart unless-stopped -p 80:8501 -p 8000:8000 radfrac:latest
```

## Security and cost notes

* The API has **no authentication** (see main README). A public demo is fine for
  coursework; don't upload real patient images to it. For anything real, put it
  behind HTTPS (ALB + ACM certificate) and an auth layer.
* Uploaded images and the SQLite history live in the `radfrac-runtime` volume
  on the instance.
* **Stop or terminate the instance after evaluation.** A t3.medium costs roughly
  USD 0.04–0.05 per hour while running.
