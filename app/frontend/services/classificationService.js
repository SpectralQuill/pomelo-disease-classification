import axios from 'axios';
import { Platform } from 'react-native';
import { API_HOST, API_PORT } from '@env';

function getBaseURL() {
  let host = API_HOST || 'localhost';
  let port = API_PORT || 5000;

  // Android Emulator: use 10.0.2.2 if host is localhost
  if (Platform.OS === 'android') {
    if (host === 'localhost' || host === '127.0.0.1') {
      host = '10.0.2.2';
    }
  }

  // iOS Simulator: use localhost (already works)
  if (Platform.OS === 'ios') {
    if (host === '0.0.0.0') host = 'localhost';
  }

  // Web (Expo Web or browser): same machine
  if (Platform.OS === 'web') {
    host = host === '0.0.0.0' ? 'localhost' : host;
  }

  return `http://${host}:${port}`;
}

const API_BASE_URL = getBaseURL();

class ClassificationService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
    });
  }

  async classifyImage(imageUri) {
    try {
      console.log(`📤 Sending image to ${API_BASE_URL}/predict`);

      const formData = new FormData();
      const filename = imageUri?.split('/')?.pop() || 'pomelo.jpg';

      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: filename,
      });

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      console.log('✅ Classification successful:', data);
      return data;
    } catch (error) {
      console.error('❌ Classification error:', error);
      if (error.message.includes('Network request failed')) {
        throw new Error(`Cannot connect to server at ${API_BASE_URL}`);
      }
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch {
      throw new Error(`Backend service is unavailable at ${API_BASE_URL}`);
    }
  }
}
const classificationService = new ClassificationService();
export default classificationService;
