import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import ClassificationService from '../services/classificationService';

const ClassificationScreen = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend status on component mount
  useEffect(() => {
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      setBackendStatus('checking');
      const status = await ClassificationService.healthCheck();
      setBackendStatus('connected');
      console.log('Backend status:', status);
    } catch (error) {
      setBackendStatus('disconnected');
      console.log('Backend error:', error.message);
    }
  };

  const pickImage = async () => {
    try {
        const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
        
        if (!permissionResult.granted) {
        Alert.alert('Permission required', 'Sorry, we need camera roll permissions!');
        return;
        }

        const pickerResult = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
        });

        console.log('Picker result:', pickerResult);

        if (!pickerResult.canceled && pickerResult.assets && pickerResult.assets.length > 0) {
        const selectedAsset = pickerResult.assets[0];
        console.log('Selected image URI:', selectedAsset.uri);
        setSelectedImage(selectedAsset.uri);
        setResult(null);
        }
    } catch (error) {
        console.error('Image picker error:', error);
        Alert.alert('Error', 'Failed to pick image');
    }
  };

  const takePhoto = async () => {
    try {
      console.log('Opening camera...');
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
      
      if (!permissionResult.granted) {
        Alert.alert('Permission required', 'Sorry, we need camera permissions to make this work!');
        return;
      }

      const cameraResult = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      console.log('Camera result:', cameraResult);
      
      if (!cameraResult.canceled && cameraResult.assets && cameraResult.assets.length > 0) {
        setSelectedImage(cameraResult.assets[0].uri);
        setResult(null);
        console.log('Photo taken:', cameraResult.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to take photo');
      console.error('Camera error:', error);
    }
  };

  const classifyImage = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select an image first');
      return;
    }

    if (backendStatus !== 'connected') {
      Alert.alert('Backend Unavailable', 'Please ensure the backend server is running and connected.');
      return;
    }

    setLoading(true);
    try {
      console.log('Starting classification...');
      const classificationResult = await ClassificationService.classifyImage(selectedImage);
      setResult(classificationResult);
      console.log('Classification result:', classificationResult);
    } catch (error) {
      Alert.alert('Classification Error', error.message);
      console.error('Classification error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = () => {
    switch (backendStatus) {
      case 'connected': return '#4CAF50';
      case 'disconnected': return '#F44336';
      default: return '#FF9800';
    }
  };

  const getStatusText = () => {
    switch (backendStatus) {
      case 'connected': return 'Backend Connected ✓';
      case 'disconnected': return 'Backend Disconnected ✗';
      default: return 'Checking Backend...';
    }
  };

  const getStatusDescription = () => {
    switch (backendStatus) {
      case 'connected': return 'Ready to classify images';
      case 'disconnected': return 'Cannot connect to backend server';
      default: return 'Checking server status...';
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <Text style={styles.title}>🍊 Pomelo Disease Classifier</Text>
      
      {/* Backend Status */}
      <View style={styles.statusSection}>
        <View style={[styles.statusContainer, { backgroundColor: getStatusColor() }]}>
          <View style={styles.statusTextContainer}>
            <Text style={styles.statusText}>{getStatusText()}</Text>
            <Text style={styles.statusDescription}>{getStatusDescription()}</Text>
          </View>
          <TouchableOpacity onPress={checkBackendStatus} style={styles.retryButton}>
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Image Display */}
      <View style={styles.imageSection}>
        <Text style={styles.sectionTitle}>Pomelo Leaf Image</Text>
        <View style={styles.imageContainer}>
          {selectedImage ? (
            <Image source={{ uri: selectedImage }} style={styles.image} />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderText}>No image selected</Text>
              <Text style={styles.placeholderSubtext}>Choose or take a photo of pomelo leaves</Text>
            </View>
          )}
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionsSection}>
        <Text style={styles.sectionTitle}>Actions</Text>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>📁 Choose from Gallery</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.button} onPress={takePhoto}>
            <Text style={styles.buttonText}>📷 Take Photo</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[
              styles.button, 
              styles.classifyButton, 
              (!selectedImage || backendStatus !== 'connected' || loading) && styles.buttonDisabled
            ]} 
            onPress={classifyImage}
            disabled={loading || !selectedImage || backendStatus !== 'connected'}
          >
            {loading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.buttonText}>Classifying...</Text>
              </View>
            ) : (
              <Text style={styles.buttonText}>🔍 Classify Disease</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>

      {/* Results Display */}
      {result && (
        <View style={styles.resultsSection}>
          <Text style={styles.sectionTitle}>Classification Results</Text>
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Prediction:</Text>
            <View style={[
              styles.predictionCard, 
              result.predicted_class === 'Healthy' ? styles.healthyCard : styles.diseaseCard
            ]}>
              <Text style={styles.predictedClass}>
                {result.predicted_class}
              </Text>
              <Text style={styles.confidence}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </Text>
            </View>
            
            {/* All predictions */}
            <Text style={styles.subTitle}>All Possibilities:</Text>
            {Object.entries(result.all_predictions)
              .sort(([,a], [,b]) => b - a)
              .map(([className, confidence]) => (
              <View key={className} style={styles.predictionRow}>
                <Text style={styles.className}>{className}</Text>
                <View style={styles.confidenceBarContainer}>
                  <View 
                    style={[
                      styles.confidenceBar, 
                      { width: `${confidence * 95}%` },
                      className === 'Healthy' ? styles.healthyBar : styles.diseaseBar
                    ]} 
                  />
                  <Text style={styles.classConfidence}>
                    {(confidence * 100).toFixed(1)}%
                  </Text>
                </View>
              </View>
            ))}
          </View>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  contentContainer: {
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#2c3e50',
  },
  statusSection: {
    marginBottom: 25,
  },
  statusContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusTextContainer: {
    flex: 1,
  },
  statusText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  statusDescription: {
    color: 'white',
    fontSize: 12,
    opacity: 0.9,
  },
  retryButton: {
    backgroundColor: 'rgba(255,255,255,0.3)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  retryText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 14,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#495057',
  },
  imageSection: {
    marginBottom: 25,
  },
  imageContainer: {
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 300,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#dee2e6',
  },
  placeholder: {
    width: 300,
    height: 300,
    backgroundColor: '#e9ecef',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#dee2e6',
    borderStyle: 'dashed',
  },
  placeholderText: {
    color: '#6c757d',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  placeholderSubtext: {
    color: '#adb5bd',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  actionsSection: {
    marginBottom: 25,
  },
  buttonContainer: {
    gap: 12,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  classifyButton: {
    backgroundColor: '#34C759',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  resultsSection: {
    marginBottom: 20,
  },
  resultContainer: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 3,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#2c3e50',
  },
  predictionCard: {
    padding: 20,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: 'center',
    borderWidth: 2,
  },
  healthyCard: {
    backgroundColor: '#d4edda',
    borderColor: '#c3e6cb',
  },
  diseaseCard: {
    backgroundColor: '#f8d7da',
    borderColor: '#f5c6cb',
  },
  predictedClass: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#155724',
  },
  confidence: {
    fontSize: 16,
    color: '#6c757d',
    fontWeight: '500',
  },
  subTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    color: '#495057',
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f3f4',
  },
  className: {
    fontSize: 16,
    flex: 1,
    color: '#495057',
  },
  confidenceBarContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
  confidenceBar: {
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  healthyBar: {
    backgroundColor: '#28a745',
  },
  diseaseBar: {
    backgroundColor: '#dc3545',
  },
  classConfidence: {
    fontSize: 14,
    fontWeight: '500',
    minWidth: 45,
    textAlign: 'right',
    color: '#6c757d',
  },
});

export default ClassificationScreen;
