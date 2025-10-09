import axios from 'axios';

const API_BASE_URL = 'http://10.0.2.2:5000';

class ClassificationService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
    });
  }

  async classifyImage(imageUri) {
    try {
      console.log('📤 Sending image for classification...');
      
      // React Native compatible approach
      const formData = new FormData();
      
      // Extract filename from URI or use default
      let filename = 'pomelo.jpg';
      if (imageUri) {
        const parts = imageUri.split('/');
        filename = parts[parts.length - 1] || 'pomelo.jpg';
      }
      
      // Append the image file in React Native compatible way
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: filename,
      });

      console.log('FormData created, sending request...');

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ Classification successful:', data);
      return data;
    } catch (error) {
      console.error('❌ Classification error:', error);
      let errorMessage = 'Failed to classify image. Please try again.';
      
      if (error.message?.includes('Network request failed')) {
        errorMessage = `Cannot connect to server at ${API_BASE_URL}. Make sure the backend is running.`;
      } else if (error.message?.includes('HTTP error')) {
        errorMessage = `Server error: ${error.message}`;
      }
      
      throw new Error(errorMessage);
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      throw new Error(`Backend service is unavailable at ${API_BASE_URL}`);
    }
  }
}

export default new ClassificationService();
