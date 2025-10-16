#!/usr/bin/env node
const { execSync } = require("child_process");
const path = require("path");
require(path.resolve(__dirname, "../app/frontend/node_modules/dotenv")).config({
  path: path.resolve(__dirname, "../.env"),
});

const containerName = process.env.DOCKER_CONTAINER_NAME || "pomelo-backend";

try {
  console.log(`🛑 Attempting to stop container '${containerName}'...`);
  execSync(`docker stop ${containerName}`, { stdio: "inherit" });
  console.log(`✅ Container '${containerName}' stopped.`);
} catch (err) {
  console.log(`⚠️ Container '${containerName}' not running or not found.`);
}
