import React, { useEffect } from 'react';
import { View, Text, StatusBar, StyleSheet } from 'react-native';
import axios from 'axios';
import Homescreen from "./screens/HomeScreen";
import AppNavigator from "./navigation/AppNavigator";
import { PaperProvider } from 'react-native-paper';
import { theme } from './theme/theme'

export default function App() {
  const [message, setMessage] = React.useState('');

  //not in use yetr
  useEffect(() => {
  axios.get('http://10.0.2.2:5000/') //ip to be used if using the emulator like android studio, else you might need to get your PC's IP address
    .then(response => {
      console.log('Response from Flask:', response.data);
      setMessage(response.data.message); // access the message string
    })
    .catch(error => {
      console.error('Error fetching from backend:', error);
    });
}, []);

  return(
    <PaperProvider theme={theme}>
      <AppNavigator/>
    </PaperProvider>
  )
}
