import React from 'react';
import {View, Text, StyleSheet, Image} from 'react-native';
import { Button } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import Layout from '../components/Layout';

const HomeScreen = () => {
  const navigation = useNavigation();

  // return (
  //   <View style={styles.container}>
  //     <Text style={styles.title}>Pomelo</Text>
  //     <Text style={styles.text}>Disease Detection</Text>
  //     <Button
  //       mode="contained"
  //       onPress={() => navigation.navigate('Scanner')}
  //       style={styles.button}
  //     >
  //       Start Scanning
  //     </Button>
  //   </View>
  // );

  //the gallery selection should immediately go to the ResultsScreen after picking an image
  
  return (
    <Layout>
      <Text>Scan Your Pomelo</Text>
      <View style={{backgroundColor: "#cfcfcfff"}}>
        <Button 
          mode='contained'
          onPress={() => navigation.navigate('Scanner')}
        >
          Open Your Camera
        </Button>

        <Text>or</Text>

        <Button 
          mode='contained'
          onPress={() => navigation.navigate('Gallery')}
        >
          Open From Gallery
        </Button>
      </View>
      <Text>From Previous Results</Text>
      <View style={{backgroundColor: "#cfcfcfff"}}>
        <Text>Currently no results</Text>
      </View>
    </Layout>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 20,
  },
  logo: {
    width: 250,
    height: 250,
  },
  title:{
    fontSize: 30,
    fontWeight: 'bold',
  },
  text:{
    fontSize: 14,
    marginBottom: 25
  },
  button: {
    marginVertical: 10,
    backgroundColor: "green",
  },
  note: {
    fontSize: 12,
    color: 'gray',
    textAlign: 'center',
    marginTop: 20,
  },
});

export default HomeScreen;
