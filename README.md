# 🌱 CropSense Web App

A Flask-based web application for presenting research results and tools related to crop analysis and plant health detection.

## 🛠️ Tech Stack

- **Backend:** Flask
- **Frontend:** HTML, Bootstrap, CSS
- **Deployment-ready:** Docker + Gunicorn

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

---

## 📦 1. Clone the repository

```bash
git clone https://github.com/AlexandraMol/cropsense-web-app.git
cd cropsense-web-app
```

---

## 🐍 2. Create a virtual environment

```bash
python -m venv .venv
```

---

## ⚙️ 3. Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Mac/Linux

```bash
source .venv/bin/activate
```

---

## 📥 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ 5. Run the application

### Option 1 (recommended)

```bash
flask --app app.py --debug run
```

### Option 2

```bash
export FLASK_APP=app.py
python -m flask run
```

Then open your browser at:

```
http://127.0.0.1:5000/
```

---

## 🐳 Running with Docker (Recommended for Deployment)

### 1. Build the Docker image

```bash
docker build -t cropsense-web-app .
```

---

### 2. Run the container

```bash
docker run -p 5000:5000 cropsense-web-app
```

---

### 🌐 Access the app

```
http://localhost:5000/
```

---

## 📁 Project Structure

```
cropsense-web-app/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── templates/
│   ├── base.html
│   ├── homepage.html
│   ├── pipeline-upload.html
│   ├── pipeline-result.html
│   └── resources.html
└── static/
    ├── css/
    │   └── global.css
    └── assets/
        └── images/
```

## 🛠️ Development Notes

- Flask renders HTML templates using Jinja2
- Static assets (CSS, images) are served from the `static/` directory
- The app is structured for simplicity and rapid prototyping
- Production deployments use **Gunicorn** inside Docker

---

## 🔄 Updating Dependencies

Whenever you install a new package, update the requirements file:

```bash
pip freeze > requirements.txt
```

---

## 🚫 Important

- Do **not** commit `.venv/`
- Make sure `.venv/` is included in `.gitignore`

---

## 👥 Contributors

- Delia Fragă

- Maria Alexandra Molnar

- Olesia Yankiv

- Antoine Herbaux

- Chiemerie David Ekweanua

---

## 📄 License

This project is for academic/research purposes.
