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
        const photoUri = cameraResult.assets[0].uri;
        console.log('Photo taken:', photoUri);
        navigation.navigate('Result', { photoUri });
      } else {
        console.log('No photos taken');
        navigation.goBack();
      }
    }
    catch (error) {
      Alert.alert('Error', 'Failed to take photo');
      console.error('Camera error:', error);
    }
    return (
      <View style={[appStyle.container, styles.center]}>
        <Text style={styles.text}>Opening camera...</Text>
      </View>
    );
  };
}

const styles = StyleSheet.create({
  center: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    fontSize: 16,
    color: '#555',
  },
});

export default ScannerScreen;