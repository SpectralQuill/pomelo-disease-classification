import React, { useEffect } from 'react';
import { View, Text, StatusBar, StyleSheet } from 'react-native';
import axios from 'axios';
import Homescreen from "./screens/HomeScreen";
import AppNavigator from "./navigation/AppNavigator";
import { PaperProvider } from 'react-native-paper';
import { theme } from './theme/theme';
import { ServerConnectionProvider } from './context/ServerConnectionContext';

export default function App() {

  return (
    <ServerConnectionProvider>
      <PaperProvider theme={theme}>
        <AppNavigator />
      </PaperProvider>
    </ServerConnectionProvider>

  )
}
