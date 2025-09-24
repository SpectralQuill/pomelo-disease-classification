import React, { useEffect } from 'react';
import {View, Image} from 'react-native';
import { appStyle } from '../theme/style';

const OpeningScreen = ({navigation}) => {
  
  useEffect(() => {
    const timer = setTimeout(() => {
        navigation.replace('Main');
    }, 2000);
    return () => clearTimeout(timer); //in case of early unmounting
  }, [navigation]);

  return (
    <View style={appStyle.container}>
      <Image source={require('../assets/app-logo.png')} style={appStyle.logo_large}/>
    </View>
  );
};

export default OpeningScreen;
