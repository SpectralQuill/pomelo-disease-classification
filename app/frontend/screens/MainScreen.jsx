import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import React from "react";
import theme from "../theme/theme";
import HomeScreen from "./HomeScreen";
import PreviousResultsScreen from "./PreviousResultsScreen";
import ProfileScreen from "./ProfileScreen";
import { SimpleLineIcons, FontAwesome, MaterialIcons } from '@expo/vector-icons'

const Tab = createBottomTabNavigator();
export default function MainScreen() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: true,
        tabBarInactiveTintColor: "#FFFFFF",
        tabBarStyle: {
          backgroundColor: theme.colors.primary,
          borderTopWidth: 0,
          elevation: 5,
          height: 120,
        },
        tabBarIcon: ({ color, size, focused }) => (
          <MaterialIcons
            name={focused ? "home" : "home"}
            color={color}
            size={size + 40}
          />
        ),
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <MaterialIcons name="home" color={color} size={30} />
          ),
        }}
      />
      <Tab.Screen
        name="Gallery"
        component={PreviousResultsScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <MaterialIcons name="photo-library" color={color} size={size} />
          ),
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <MaterialIcons name="person" color={color} size={size} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}