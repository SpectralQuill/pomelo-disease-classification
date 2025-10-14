#!/usr/bin/env node
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");

// ------------------------------------------
// 🧩 Load environment variables
// ------------------------------------------
require(path.resolve(__dirname, "../app/frontend/node_modules/dotenv")).config({
  path: path.resolve(__dirname, "../.env"),
});

// ------------------------------------------
// ⚙️ Environment setup
// ------------------------------------------
const imageName = process.env.DOCKER_IMAGE_NAME || "pomelo-backend";
const containerName = process.env.DOCKER_CONTAINER_NAME || "pomelo-backend";
const flaskHost = process.env.FLASK_HOST || "0.0.0.0";
const flaskPort = process.env.FLASK_PORT || "5000";
const hostPort = process.env.HOST_PORT || "8080";
const backendDir = path
  .resolve(__dirname, "../app/backend")
  .replace(/\\/g, "/");
const reqFile = path.join(backendDir, "requirements.txt");
const hashFilePath = path.resolve(__dirname, "../.backend-hash.json");

// ------------------------------------------
// 🧠 Helper functions
// ------------------------------------------
function run(command, exitOnFail = true) {
  try {
    console.log(`\n▶️ ${command}`);
    execSync(command, { stdio: "inherit", shell: true });
  } catch (err) {
    console.error(`❌ Command failed: ${command}`);
    if (exitOnFail) process.exit(1);
  }
}

function hashFile(filepath) {
  const buffer = fs.readFileSync(filepath);
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function hashFolder(dir) {
  const files = [];
  function walk(current) {
    for (const entry of fs.readdirSync(current)) {
      const full = path.join(current, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) walk(full);
      else if (stat.isFile()) files.push(full);
    }
  }
  walk(dir);
  const hash = crypto.createHash("sha256");
  for (const file of files) {
    hash.update(fs.readFileSync(file));
  }
  return hash.digest("hex");
}

function dockerExists(type, name) {
  try {
    execSync(`docker ${type} inspect "${name}"`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

// ------------------------------------------
// 🔍 Step 1: Detect backend & requirements changes
// ------------------------------------------
console.log("🔍 Checking backend and requirements for changes...");

const currentBackendHash = hashFolder(backendDir);
const currentReqHash = fs.existsSync(reqFile) ? hashFile(reqFile) : null;

let previous = {};
if (fs.existsSync(hashFilePath)) {
  previous = JSON.parse(fs.readFileSync(hashFilePath, "utf8"));
}

const backendChanged = previous.backendHash !== currentBackendHash;
const reqChanged = previous.reqHash !== currentReqHash;

console.log(`📦 requirements.txt changed: ${reqChanged}`);
console.log(`📂 backend files changed: ${backendChanged}`);

// ------------------------------------------
// 🧱 Step 2: Check image & container state
// ------------------------------------------
const imageExists = dockerExists("image", imageName);
const containerExists = dockerExists("container", containerName);

// ------------------------------------------
// 🧱 Step 3: Decide build/recreate logic
// ------------------------------------------
if (!imageExists || backendChanged || reqChanged) {
  if (reqChanged) console.log("📦 requirements.txt changed — full rebuild needed.");
  else if (backendChanged) console.log("🧩 Backend code changed — rebuilding image.");
  else console.log("🛠️ Image missing — building fresh image.");

  // Stop & remove old container
  if (containerExists) {
    console.log(`🧹 Removing existing container '${containerName}'...`);
    run(`docker stop "${containerName}"`, false);
    run(`docker rm "${containerName}"`, false);
  }

  // Build (Docker caching keeps pip layers unless requirements.txt changed)
  run(`docker build -t "${imageName}" "${backendDir}"`);

  // Create new container
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
  run(createCmd);

  // Update stored hash
  fs.writeFileSync(
    hashFilePath,
    JSON.stringify(
      {
        backendHash: currentBackendHash,
        reqHash: currentReqHash,
      },
      null,
      2
    )
  );
} else {
  console.log("✅ No changes detected — reusing existing image and container.");
  if (!containerExists) {
    console.log(`⚙️ Creating missing container '${containerName}'...`);
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
    run(createCmd);
  }
}

// ------------------------------------------
// ▶️ Step 4: Start container (guaranteed)
// ------------------------------------------
console.log(`▶️ Starting container '${containerName}'...`);
run(`docker start -a "${containerName}"`);
