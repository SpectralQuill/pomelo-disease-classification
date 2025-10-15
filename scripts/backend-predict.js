#!/usr/bin/env node
const { execSync } = require("child_process");
const path = require("path");

// Use dotenv from frontend’s node_modules
require(path.resolve(__dirname, "../app/frontend/node_modules/dotenv")).config({
  path: path.resolve(__dirname, "../.env"),
});

const image = process.argv[2];
if (!image) {
  console.error("❌ Missing image path.\nUsage: npm run backend:predict <path_to_image>");
  process.exit(1);
}

const host = process.env.API_HOST || "localhost";
const port = process.env.API_PORT || process.env.HOST_PORT || 8080;
const url = `http://${host}:${port}/predict`;

console.log(`📤 Sending ${image} to ${url}`);

try {
  execSync(`curl -X POST -F "image=@${image}" ${url}`, { stdio: "inherit", shell: true });
} catch (err) {
  console.error("❌ Prediction failed.");
  process.exit(1);
}
