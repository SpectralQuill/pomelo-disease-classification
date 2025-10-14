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

const host = process.env.FLASK_HOST || "127.0.0.1";
const port = process.env.FLASK_PORT || 5000;
const url = `http://${host}:${port}/predict`;

console.log(`📤 [Backend] Sending ${image} directly to ${url}`);

try {
  execSync(`curl -X POST -F "image=@${image}" ${url}`, { stdio: "inherit", shell: true });
} catch (err) {
  console.error("❌ Direct backend prediction failed.");
  process.exit(1);
}
