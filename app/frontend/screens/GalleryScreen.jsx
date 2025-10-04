import React, { useState, useEffect } from "react";
import { View, Image, StyleSheet } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { useNavigation } from "@react-navigation/native";

const GalleryScreen = () => {
  const navigation = useNavigation();

  useEffect(() => {
    pickImage();
  }, []);

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      alert("Permission for gallery is required");
      navigation.goBack(); // go back if denied
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      navigation.navigate("Result", { imageUri: uri });
    } else {
      navigation.goBack(); 
    }
  };

  return <View style={styles.container}></View>;
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff" },
});

export default GalleryScreen;