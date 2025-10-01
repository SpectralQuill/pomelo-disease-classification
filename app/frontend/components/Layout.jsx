import React from "react";
import { appStyle } from "../theme/style";
import { View, Image } from "react-native";
import { Appbar } from "react-native-paper";
import  theme  from '../theme/theme'
import AppHeader from "./AppHeader";


const Layout = ({children}) => {
    return (
        <View style={appStyle.container}>
            <AppHeader/>
            
            {/*This is where the other screen components gets rendered*/}
            <View style={appStyle.screen}>
                {children}
            </View>
        </View>
    )
}

export default Layout;