import React from 'react';
import {View, Text, StyleSheet, Image} from 'react-native';
import { Button } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import Layout from '../components/Layout';
import theme from '../theme/theme';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';

const HomeScreen = () => {
  const navigation = useNavigation();  
  return (
    <Layout>
      <Text style={{color: theme.colors.primary, fontSize: 18}}>Scan Your Pomelo</Text>
      <View style={{backgroundColor: "#eeeeeeff", alignItems: "center", height: 250,justifyContent: "center", borderRadius: 10,
        paddingBottom: 10, paddingTop: 10, paddingRight: 30, paddingLeft: 30,
      }}>
        <Button
          mode="contained"
          style={{
            width: "100%",
            backgroundColor: theme.colors.primary,
          }}
          contentStyle={{
            height: 60,
            justifyContent: "center",
          }}
          onPress={() => navigation.navigate("Scanner")}
        >
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <MaterialIcons name="photo-camera" color="#fff" size={26} />
            <Text style={{ color: "#fff", marginLeft: 8, fontSize: 20}}>Scan from Camera</Text>
          </View>
        </Button>

        <Text style={{alignItems: "center", marginBottom: 20, marginTop: 20, fontSize: 20}}>or</Text>
        
        <Button
          mode="contained"
          style={{
            width: "100%",
            backgroundColor: theme.colors.primary,
          }}
          contentStyle={{
            height: 60,
            justifyContent: "center",
          }}
          onPress={() => navigation.navigate("Gallery")}
        >
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <MaterialIcons name="photo-library" color="#fff" size={26} />
            <Text style={{ color: "#fff", marginLeft: 8, fontSize: 20 }}>Open from Gallery</Text>
          </View>
        </Button>
      </View>

      {/* This layout will require me a database to establish first, only finish if main priority is finish <possible>*/}

      {/* <Text>From Previous Results</Text>
      <View style={{backgroundColor: "#cfcfcfff"}}>
        <Text>Currently no results</Text>
      </View> */}
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
