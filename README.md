# HyperSoilNet - Full Stack Deployment

This repository contains the complete system for generating quantitative soil fertility predictions (B, Fe, Zn, Cu, Mn, S) from HYPERVIEW2 airborne hyperspectral images (`.npz`).

The architecture is split into two parts:
1. **Frontend**: A static HTML/CSS/JS dashboard that handles `.npz` file uploads.
2. **Backend API (`/hypesoilnet-api`)**: A FastAPI Python server that runs the robust CNN and XGBoost/CatBoost/RF ensemble inference.

## How to Host on a New Device

If you want to host this on a different local machine, edge device (like a Raspberry Pi or local tower), or private lab server, you must run both the Frontend and the Backend locally.

### Step 1: Run the Backend API

The backend loads the heavily processed machine learning models (`.pkl` and `.pt`). You can run it either via Python natively or via Docker.

**Option A: Using Python (Recommended)**
Open a terminal in the `hypesoilnet-api` folder and install the exact mathematical libraries required to prevent model collision:
```bash
cd hypesoilnet-api
pip install -r requirements.txt
```
Then, launch the FastAPI server:
```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```
*The API is now listening at `http://localhost:7860/predict`*

**Option B: Using Docker**
If the device has Docker installed, you can build and run the contained image without worrying about Python versions:
```bash
cd hypesoilnet-api
docker build -t hypersoilnet-api .
docker run -p 7860:7860 hypersoilnet-api
```

### Step 2: Configure the Frontend

Because the Backend is now running locally on your device (at `http://localhost:7860`), we need to tell the Dashboard where to send the `.npz` files instead of sending them to Hugging Face.

1. Open `script.js` in a text editor.
2. At the top of the file, find this line:
   ```javascript
   const API_URL = 'https://amsh4-hypesoilnet.hf.space/predict';
   ```
3. Change it to your new local address:
   ```javascript
   const API_URL = 'http://localhost:7860/predict';
   ```
*(Note: If you are hosting the API on a network server instead of localhost, replace `localhost` with the server's local IP address like `192.168.1.100`)*

### Step 3: Run the Frontend

Simply open `index.html` in your web browser! 
Alternatively, use the VS Code "Live Server" extension, or run a simple python server in the root of this folder:
```bash
python -m http.server 8000
```
Navigate to `http://localhost:8000/index.html` in your browser. Drag and drop a `.npz` file, and you will see the full prediction pipeline run successfully on your local device.
