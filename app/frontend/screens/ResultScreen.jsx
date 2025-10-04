import { View, Text, StyleSheet, Image } from 'react-native';
import AppHeader from '../components/AppHeader';
import { appStyle } from '../theme/style';
import { useNavigation } from '@react-navigation/native';
import { Button } from 'react-native-paper';
import { useState } from 'react';
import SelectionModal from '../components/modals/SelectionModal';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';

const ResultScreen = ({ route }) => {
  const navigation = useNavigation();
  const [showSelectModal, setShowSelectModal] = useState(false);
  const { photoUri } = route.params || {};

  //Result screen have a different layout than the rest, so it should only have the header
  return (
    <View style={appStyle.container}>
      <AppHeader/>
      <View style={{backgroundColor: "#eeeeeeff", alignItems: "center", flex:1, justifyContent: "center", borderRadius: 10,
        paddingBottom: 10, paddingTop: 10, paddingRight: 30, paddingLeft: 30,
      }}>
        
        {photoUri ? (
          <Image source={{ uri: photoUri }} style={styles.image} />
        ) : (
          <Text>No image captured</Text>
        )}
        <Text style={styles.title}>Result</Text>
        <Text>Description of scanned pomelo goes here</Text>

        {/*button containers for retry and home*/}
        <View style={{width:'100%', height: 100}}>
          <Button
            mode="text"
            contentStyle={{
              height: 60,
              justifyContent: "center",
            }}
            onPress={() => setShowSelectModal(!showSelectModal ? true: false)}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <MaterialIcons name="photo-camera" color="#fff" size={26} />
              <Text style={{ color: "#fff", marginLeft: 8, fontSize: 20}}>Retry</Text>
            </View>
          </Button>
          <Button
            mode="text"
            contentStyle={{
              height: 60,
              justifyContent: "center",
            }}
            onPress={() => navigation.navigate("Main")}
          >
            <MaterialIcons name="home" color="#fff" size={26} />
            <Text style={{ color: "#fff", marginLeft: 8, fontSize: 20}}>Home</Text>
          </Button>
        </View>
      </View>
      {showSelectModal && <SelectionModal isVisible={showSelectModal}></SelectionModal>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 10 },
  image: { width: 250, height: 250, marginVertical: 10 },
});

export default ResultScreen;
