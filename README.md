# Workflow-CI — Heart Disease ML System

Repository ini berisi MLflow Project dan GitHub Actions CI/CD pipeline untuk melatih model Heart Disease secara otomatis dan mendeploy ke Docker Hub.

---

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                         # GitHub Actions CI pipeline
├── MLProject/
│   ├── MLProject                          # Konfigurasi MLflow Project
│   ├── conda.yaml                         # Environment dependencies
│   ├── modelling.py                       # Script training model
│   └── heart_disease_preprocessing.csv   # Dataset siap latih
└── heart_disease_preprocessing.csv
```

---

## Tentang Project

| Info | Detail |
|------|--------|
| Dataset | Heart Disease UCI (302 baris, 13 fitur) |
| Model | Random Forest Classifier |
| Tracking | MLflow + DagsHub (online) |
| CI/CD | GitHub Actions |
| Container | Docker Hub |

---

## Alur CI/CD Pipeline

Setiap kali ada `push` ke branch `main`, GitHub Actions otomatis menjalankan:

```
Push ke GitHub
      ↓
1. Setup Python 3.12.7
      ↓
2. Install dependencies
      ↓
3. Set MLflow Tracking URI → DagsHub
      ↓
4. Jalankan MLflow Project (training model)
      ↓
5. Ambil run_id terbaru dari DagsHub
      ↓
6. Upload artefak ke GitHub
      ↓
7. Build Docker image dari model
      ↓
8. Push Docker image ke Docker Hub
```

---

## Hasil Pipeline

Setelah pipeline berhasil jalan:

- **Eksperimen** tersimpan di DagsHub → tab Experiments
- **Artefak** (confusion matrix, model) tersimpan di GitHub Actions artifacts
- **Docker image** tersimpan di Docker Hub dengan tag `latest`

---

## Cara Menjalankan Ulang Pipeline

Pipeline otomatis jalan setiap `push`. Untuk trigger manual:

1. Buka tab **Actions** di GitHub
2. Klik workflow **"MLflow CI Pipeline - Heart Disease"**
3. Klik tombol **"Run workflow"**
4. Klik **"Run workflow"** (hijau)

---

## Cara Pull Docker Image

Setelah pipeline berhasil, image bisa langsung dipakai:

```bash
# Pull image
docker pull shazidian/heart-disease-model:latest

# Jalankan container
docker run -p 5001:8080 shazidian/heart-disease-model:latest
```

---

## GitHub Secrets yang Dibutuhkan

Pastikan secrets berikut sudah diset di **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|--------|------------|
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_TOKEN` | Access token DagsHub |
| `DOCKER_USERNAME` | Username Docker Hub |
| `DOCKER_PASSWORD` | Password Docker Hub |

---

## Metrik Model

| Metrik | Nilai |
|--------|-------|
| Accuracy | ~0.75 |
| F1 Score | ~0.78 |
| Precision | ~0.76 |
| Recall | ~0.79 |
| ROC AUC | ~0.86 |

---

## Author

**shazidian** — Submission MSML Dicoding
