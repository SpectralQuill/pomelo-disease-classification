import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { View } from "react-native";
import theme from "../theme/theme";
import HomeScreen from "./HomeScreen";
import PreviousResultsScreen from "./PreviousResultsScreen";
import ProfileScreen from "./ProfileScreen";
import { MaterialIcons } from '@expo/vector-icons'

const Tab = createBottomTabNavigator();
export default function MainScreen() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: true,
        tabBarInactiveTintColor: theme.colors.primary,
        tabBarActiveTintColor: '#fff',
        tabBarStyle: {
          borderTopWidth: 0,
          elevation: 5,
          height: 120,
        },
        tabBarItemStyle: {
          paddingTop: 10
        },
        tabBarLabelStyle: {
          fontSize: 12,
          marginTop: 5,
        },
      }}
    >
      {[
        { name: "Home", icon: "home", component: HomeScreen },
        { name: "Catalogue", icon: "photo-library", component: PreviousResultsScreen },
        { name: "Profile", icon: "person", component: ProfileScreen },
      ].map((tab) => (
        <Tab.Screen
          key={tab.name}
          name={tab.name}
          component={tab.component}
          options={{
            tabBarIcon: ({ focused }) => (
              <View
                style={{
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <View
                  style={{
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: focused ? theme.colors.primary : "transparent",
                    width: 90,
                    height: 90,
                    borderRadius: 50,
                    elevation: focused ? 4 : 0,
                    shadowColor: "#000",
                    shadowOpacity: 0.25,
                    shadowRadius: 4,
                    shadowOffset: { width: 0, height: 2 },
                    bottom: 0,
                  }}
                >
                  <MaterialIcons
                    name={tab.icon}
                    size={focused ? 42 : 36}
                    color={focused ? "#fff" : theme.colors.primary}
                  />
                </View>
              </View>
            ),
          }}
        />
      ))}
    </Tab.Navigator>
  );
}