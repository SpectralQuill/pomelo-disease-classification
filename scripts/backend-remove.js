#!/usr/bin/env node
const { execSync } = require("child_process");
const path = require("path");
require(path.resolve(__dirname, "../app/frontend/node_modules/dotenv")).config({
  path: path.resolve(__dirname, "../.env"),
});

const imageName = process.env.DOCKER_IMAGE_NAME || "pomelo-backend";
const containerName = process.env.DOCKER_CONTAINER_NAME || "pomelo-backend";

try {
  console.log(`🧹 Removing all containers based on image '${imageName}'...`);
  execSync(
    `docker ps -a --filter "ancestor=${imageName}" --format "{{.ID}}" | xargs -r docker stop`,
    { stdio: "inherit", shell: true }
  );
  execSync(
    `docker ps -a --filter "ancestor=${imageName}" --format "{{.ID}}" | xargs -r docker rm`,
    { stdio: "inherit", shell: true }
  );

  console.log(`🗑️ Removing named container '${containerName}' (if exists)...`);
  execSync(`docker rm -f ${containerName} || true`, {
    stdio: "inherit",
    shell: true,
  });

  console.log("✅ Cleanup complete.");
} catch (err) {
  console.error("⚠️ Cleanup encountered issues:", err.message);
}
