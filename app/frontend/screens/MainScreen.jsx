import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import React from "react";
import theme from "../theme/theme";
import HomeScreen from "./HomeScreen";
import PreviousResultsScreen from "./PreviousResultsScreen";
import ProfileScreen from "./ProfileScreen";
import MaterialIcons from "@expo/vector-icons/MaterialIcons";

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
          height: 100,
        },
        tabBarIcon: ({ color, size, focused }) => (
          <MaterialIcons
            name={focused ? "home" : "home"} 
            color={color}
            size={size + 20}
          />
        ),
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <MaterialIcons name="home" color={color} size={size} />
          ),
        }}
      />
      <Tab.Screen
        name="PreviousResults"
        component={PreviousResultsScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <MaterialIcons name="person" color={color} size={size} />
          ),
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarIcon: ({ color, size }) => (
            <MaterialIcons name="image-multiple-outline" color={color} size={size} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}