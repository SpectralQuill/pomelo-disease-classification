#!/usr/bin/env node
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
require(path.resolve(__dirname, "../app/frontend/node_modules/dotenv")).config({
  path: path.resolve(__dirname, "../.env"),
});

// ---------- ENV CONFIG ----------
const imageName = process.env.DOCKER_IMAGE_NAME || "pomelo-backend";
const containerName = process.env.DOCKER_CONTAINER_NAME || "pomelo-backend";
const flaskPort = process.env.FLASK_PORT || 5000;
const hostPort = process.env.HOST_PORT || 8080;
const backendDir = path.resolve(__dirname, "../app/backend");
const cacheFile = path.resolve(__dirname, "./.docker-cache.json");

// ---------- HELPERS ----------
function hashFile(filePath) {
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath);
  return crypto.createHash("sha256").update(content).digest("hex");
}

function readCache() {
  if (!fs.existsSync(cacheFile)) return {};
  try {
    return JSON.parse(fs.readFileSync(cacheFile, "utf8"));
  } catch {
    return {};
  }
}

function writeCache(data) {
  fs.writeFileSync(cacheFile, JSON.stringify(data, null, 2));
}

// ---------- MAIN ----------
(async () => {
  try {
    console.log("🚀 Starting backend container...");

    const reqFile = path.join(backendDir, "requirements.txt");
    const currentHash = hashFile(reqFile);
    const cache = readCache();

    // 1️⃣ Check image existence
    let imageExists = true;
    try {
      execSync(`docker image inspect ${imageName}`, { stdio: "ignore" });
    } catch {
      imageExists = false;
    }

    const shouldRebuild =
      !imageExists || cache.requirementsHash !== currentHash;

    if (shouldRebuild) {
      console.log(
        imageExists
          ? "🔁 Requirements changed — rebuilding image..."
          : `🛠️ Building Docker image '${imageName}'...`
      );
      execSync(`docker build -t ${imageName} "${backendDir}"`, {
        stdio: "inherit",
      });
      cache.requirementsHash = currentHash;
      writeCache(cache);
    } else {
      console.log(`✅ Using cached image '${imageName}'`);
    }

    // 2️⃣ Check container existence properly
    const checkContainerCmd = `docker ps -a --filter "name=^/${containerName}$" --format "{{.Names}}"`;
    const containerCheck = execSync(checkContainerCmd, { encoding: "utf8" }).trim();
    const containerExists = containerCheck === containerName;

    if (!containerExists) {
      console.log(`📦 Creating container '${containerName}'...`);
      execSync(
        `docker create --name ${containerName} \
    -p ${hostPort}:${flaskPort} \
    -v "${backendDir.replace(/\\/g, "/")}:/app" \
    ${imageName}`,
        { stdio: "inherit", shell: true }
      );
    } else {
      console.log(`✅ Container '${containerName}' already exists.`);
    }

    // 3️⃣ Start container safely (auto-recreate if missing)
    console.log(`▶️ Starting container '${containerName}' on port ${hostPort}...`);
    try {
      execSync(`docker start -a ${containerName}`, { stdio: "inherit" });
    } catch (err) {
      console.log(`⚠️ Container '${containerName}' missing — recreating...`);
      execSync(`docker rm -f ${containerName} || true`, { stdio: "ignore", shell: true });
      execSync(
        `docker create --name ${containerName} \
          -p ${hostPort}:${flaskPort} \
          -v ${backendDir}:/app \
          ${imageName}`,
        { stdio: "inherit", shell: true }
      );
      execSync(`docker start -a ${containerName}`, { stdio: "inherit" });
    }

  } catch (err) {
    console.error("❌ Failed to start backend:");
    console.error(err.message);
    process.exit(1);
  }
})();
