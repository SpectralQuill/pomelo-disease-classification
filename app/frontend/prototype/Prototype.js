import React, { useEffect } from 'react';
import { View, Text, StatusBar, StyleSheet } from 'react-native';
import axios from 'axios';
import Homescreen from "../screens/HomeScreen";
import AppNavigator from "../navigation/AppNavigator";
import { PaperProvider } from 'react-native-paper';
import { theme } from '../theme/theme'
import ClassificationScreen from './ClassificationScreen';

export default function App() {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8f9fa" />
      <ClassificationScreen />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
});
