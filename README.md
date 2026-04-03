# 🌱 CropSense Web App

A Flask-based web application for presenting research results and tools related to crop analysis and plant health detection.

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

## 🛠️ Development Notes

- The app uses **Flask** for the backend
- Frontend is built with **HTML + Bootstrap + custom CSS**
- Static files are located in the `static/` folder
- Templates are located in the `templates/` folder

---

## 📁 Project Structure

```
cropsense-web-app/
│
├── app.py
├── requirements.txt
├── templates/
│   ├── homepage.html
│   ├── results.html
│   └── ...
└── static/
    ├── css/
    └── assets/
```

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

- TO BE ADDED

---

## 📄 License

This project is for academic/research purposes.
