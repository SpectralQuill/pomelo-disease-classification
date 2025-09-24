import React from "react";
import { appStyle } from "../theme/style";
import { View, Image } from "react-native";
import { Appbar } from "react-native-paper";
import  theme  from '../theme/theme'

const Layout = ({children}) => {
    return (
        <View style={appStyle.container}>
            <Appbar.Header style={{backgroundColor: theme.colors.primary}}>
                <View style={appStyle.appBar}>
                    <Image
                        source={require('../assets/app-logo-white.png')}
                        style={{ width: 50, height: 50, marginRight: 8 }}
                        resizeMode="contain"
                    />
                    <Appbar.Content title = "POMELO" titleStyle={appStyle.logo_title}/>
                </View>
            </Appbar.Header>
            {/*This is where the other screen components gets rendered*/}
            <View style={appStyle.screen}>
                {children}
            </View>
        </View>
    )
}

export default Layout;