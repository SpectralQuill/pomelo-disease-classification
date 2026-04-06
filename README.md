# 🍊 Pomelo Disease Classification App

By:
- Agawin, Sylvann Jules A.
- Apostol, Gian Tristian G.
- Cabangbang, R-Man Rey S.

This repository contains a Pomelo Disease Classification system with a **React Native (Expo) frontend** and a **Dockerized backend**. The backend serves the trained model via an API, while the frontend allows users to interact with the app on mobile.

> ⚠️ Note: This README focuses only on running the app locally (frontend + backend), not on model training.

---

## 🚀 Getting Started

Follow these steps to run the application on your local machine.

---

## 📦 Prerequisites

Make sure you have the following installed:

- **Node.js** (v16 or higher recommended)
- **npm** (comes with Node.js)
- **Docker Desktop** (must be running)
- **Expo Go** app on your mobile device (Android/iOS)

---

## ⚙️ Installation & Setup

### 1. Install Dependencies

Run the following command to install all required Node packages:

```bash
npm run install
```

---

### 2. Configure Environment Variables

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Open `.env` and update the following values:

- `API_HOST` → Your local IP address (e.g., `192.168.x.x`)
- `API_PORT` → (optional) Change if needed
- `FLASK_HOST` → (optional)
- `HOST_PORT` → (optional)

> 💡 Your mobile device and computer must be on the same network.

---

### 3. Start Docker

Make sure **Docker Desktop is running** before proceeding.

---

## 🐳 Running the Backend

Start the Dockerized backend:

```bash
npm run backend:start
```

This will:
- Build the Docker container (if not already built)
- Start the backend API service

---

## 📱 Running the Frontend (Expo)

Start the React Native Expo app:

```bash
npm run frontend:start
```

After running:
- A QR code will appear in the terminal or browser
- Open **Expo Go** on your phone
- Scan the QR code to launch the app

---

## 📲 Using the App

Once the app is running on your mobile device:

### 1. Select or Capture an Image
- Use your device camera to capture a new image, or
- Choose an existing image from your gallery
### 2. Crop the Image
- Adjust the crop area before submission
- Cropping is limited to a square aspect ratio
### 3. Submit for Classification
- After cropping, submit the image
- The app will send the image to the backend for processing
### 4. Wait for Processing
- The backend will analyze the image using the trained model
- Processing may take a few seconds depending on system performance
### 5. View Results
- The app will display:
- Predicted disease classification
- Description or details about the detected condition

## 🔄 Summary of Commands

```bash
npm run install         # Install dependencies
npm run backend:start   # Start Docker backend
npm run frontend:start  # Start Expo frontend
```

---

## 🛠 Troubleshooting

- Ensure your `.env` IP matches your local network IP
- Make sure Docker Desktop is running
- Ensure your phone and computer are connected to the same Wi-Fi network
- Restart Expo if the QR code fails to load

---

## 📄 License

This project is for academic/research purposes.

---