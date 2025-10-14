#!/usr/bin/env node
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

// ------------------------------------------
// 🧩 Load environment variables
// ------------------------------------------
require(path.resolve(__dirname, "../app/frontend/node_modules/dotenv")).config({
  path: path.resolve(__dirname, "../.env"),
});

// ------------------------------------------
// 🧱 Environment variables
// ------------------------------------------
const imageName = process.env.DOCKER_IMAGE_NAME || "pomelo-backend";
const containerName = process.env.DOCKER_CONTAINER_NAME || "pomelo-backend";
const flaskHost = process.env.FLASK_HOST || "0.0.0.0";
const flaskPort = process.env.FLASK_PORT || "5000";
const hostPort = process.env.HOST_PORT || "8080";

// Normalize backend directory to Docker-friendly format
const backendDir = path
  .resolve(__dirname, "../app/backend")
  .replace(/\\/g, "/");

// ------------------------------------------
// 🧩 Helper: Run commands safely
// ------------------------------------------
function runCommand(command, options = {}) {
  try {
    console.log(`\n▶️ ${command}`);
    execSync(command, { stdio: "inherit", shell: true, ...options });
  } catch (err) {
    console.error(`❌ Command failed: ${command}`);
    console.error(err.message);
    process.exit(1);
  }
}

// ------------------------------------------
// 🔍 Check for Docker installation
// ------------------------------------------
try {
  execSync("docker --version", { stdio: "ignore" });
} catch {
  console.error("❌ Docker is not installed or not in PATH.");
  process.exit(1);
}

// ------------------------------------------
// 🛠️ Build Docker image if missing
// ------------------------------------------
function ensureDockerImage() {
  try {
    execSync(`docker image inspect "${imageName}"`, { stdio: "ignore" });
    console.log(`✅ Docker image '${imageName}' already exists.`);
  } catch {
    console.log(`🛠️ Building Docker image '${imageName}'...`);
    runCommand(`docker build -t "${imageName}" "${backendDir}"`);
  }
}

// ------------------------------------------
// 🧱 Create Docker container if missing
// ------------------------------------------
function ensureDockerContainer() {
  try {
    execSync(`docker inspect "${containerName}"`, { stdio: "ignore" });
    console.log(`✅ Container '${containerName}' already exists.`);
  } catch {
    console.log(`🚀 Creating new container '${containerName}'...`);
    const createCmd = [
      "docker create",
      `--name "${containerName}"`,
      `-p "${hostPort}:${flaskPort}"`,
      `-v "${backendDir}:/app"`,
      `-e "FLASK_RUN_HOST=${flaskHost}"`,
      `-e "FLASK_RUN_PORT=${flaskPort}"`,
      `-e "PYTHONUNBUFFERED=1"`,
      `"${imageName}"`,
    ].join(" ");
    runCommand(createCmd);
  }
}

// ------------------------------------------
// ▶️ Start the container
// ------------------------------------------
function startDockerContainer() {
  console.log(`▶️ Starting container '${containerName}' on port ${hostPort}...`);
  try {
    runCommand(`docker start -a "${containerName}"`);
  } catch (err) {
    console.error("❌ Failed to start backend container.");
    console.error(err.message);
    process.exit(1);
  }
}

// ------------------------------------------
// 🧼 Clean up stopped containers (optional helper)
// ------------------------------------------
function removeDanglingContainers() {
  try {
    const result = execSync(`docker ps -aq -f "status=exited" -f "ancestor=${imageName}"`, {
      encoding: "utf8",
    }).trim();
    if (result) {
      console.log("🧹 Removing exited containers related to the image...");
      runCommand(`docker rm ${result}`);
    }
  } catch (_) {
    // Safe to ignore
  }
}

// ------------------------------------------
// 🚦 Execution Flow
// ------------------------------------------
console.log("🚀 Starting backend container...");

removeDanglingContainers();
ensureDockerImage();
ensureDockerContainer();
startDockerContainer();
