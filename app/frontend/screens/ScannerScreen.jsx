import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Icon, IconButton } from "react-native-paper";
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { appStyle } from '../theme/style';

export default function ScannerScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraFacing, setCameraFacing] = useState("back");
  const [flashMode, setFlashMode] = useState('off');
  const cameraRef = useRef(null);

  if (!permission) {
    return <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Requesting permissions...</Text>
    </View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text>No access to camera</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      console.log('Captured photo:', photo.uri);
      navigation.navigate('Result', { photoUri: photo.uri });
    }
  };

  const handleCameraSwitching = () => {
    setCameraFacing((prev) => (prev === "front" ? "back" : "front"));
  };

  const handleToggleFlash = () => {
    setFlashMode((prev) => {
      if (prev === 'off') return 'on';
      if (prev === 'on') return 'auto';
      if (prev === 'auto') return 'torch';
      return 'off';
    });
  }

  return (
    <View style={appStyle.container}>
      <CameraView ref={cameraRef} style={appStyle.camera_container} facing={cameraFacing} flash={flashMode} />
      <View style={appStyle.button_container}>
        <IconButton
          icon='flash'
          iconColor='white'
          size={30}
          onPress={handleToggleFlash}
        />
        <IconButton
          icon="camera"
          containerColor="green"
          iconColor="white"
          size={45}
          onPress={takePicture} />
        <IconButton
          icon={() => (
            <MaterialIcons name='cameraswitch' size={28} color={'#fff'} />
          )}
          iconColor="white"
          size={25}
          onPress={handleCameraSwitching}
        />
      </View>
    </View>
  );
}