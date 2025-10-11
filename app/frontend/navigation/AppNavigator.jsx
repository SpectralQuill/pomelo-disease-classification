import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import ResultScreen from '../screens/ResultScreen';
import ScannerScreen from '../screens/ScannerScreen';
import OpeningScreen from '../screens/OpeningScreen';
import MainScreen from '../screens/MainScreen';
import GalleryPickerScreen from '../screens/GalleryPickerScreen';

const Stack = createNativeStackNavigator();

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Opening" screenOptions={{ headerShown: false }}>
        <Stack.Screen
          name="Opening"
          component={OpeningScreen}
        />
        <Stack.Screen
          name="Main"
          component={MainScreen}
        />
        <Stack.Screen
          name="Picker"
          component={GalleryPickerScreen}
        />
        <Stack.Screen
          name="Scanner"
          component={ScannerScreen}
          options={{ title: 'Scanner' }}
        />
        <Stack.Screen
          name="Result"
          component={ResultScreen}
          options={{ title: 'Result' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
