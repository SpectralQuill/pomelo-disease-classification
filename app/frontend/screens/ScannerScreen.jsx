import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Button, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Icon, IconButton } from "react-native-paper";
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { appStyle } from '../theme/style';
import { useNavigation } from "@react-navigation/native";
import * as ImagePicker from 'expo-image-picker';

const ScannerScreen = () => {
  const navigation = useNavigation();

  useEffect(() => {
    takePicture();
  }, []);

  const takePicture = async () => {
    try {
      console.log('Opening camera...');
      const permission = await ImagePicker.requestCameraPermissionsAsync();

      if (!permission.granted) {
        Alert.alert('Permission required', 'Sorry, we need camera permissions to make this work!');
        return;
      }
      const cameraResult = await ImagePicker.launchCameraAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });
      console.log('Camera result:', cameraResult);

      if (!cameraResult.canceled && cameraResult.assets && cameraResult.assets.length > 0) {
        console.log('Photo taken:', cameraResult.assets[0].uri);
        navigation.navigate('Result', { photoUri: cameraResult.assets[0].uri });
      }
    }
    catch (error) {
      Alert.alert('Error', 'Failed to take photo');
      console.error('Camera error:', error);
    }
  };
}

export default ScannerScreen;