import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {IconButton} from "react-native-paper";

export default function ScannerScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraFacing, setCameraFacing] = React.useState("back");
  const cameraRef = useRef(null);

  if (!permission) {
    return <View><Text>Requesting permissions...</Text></View>;
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
      navigation.navigate('Result', { imageUri: photo.uri });
    }
  };

  const handleCameraSwitching = () => {
    setCameraFacing((prev) => (prev === "front" ? "back" : "front"));
  };

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing = {cameraFacing} />
      <View style={styles.buttonContainer}>
        <IconButton
            icon="camera"
            containerColor="green"
            iconColor="white"
            size={45}
            onPress={takePicture} />
        <IconButton
            icon="camera-switch"
            iconColor="white"
            size={20}
            onPress={handleCameraSwitching}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  buttonContainer: {
    flexDirection: 'row',
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
  },
});
