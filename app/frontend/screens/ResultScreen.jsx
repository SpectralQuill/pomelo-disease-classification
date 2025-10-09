import { View, ScrollView, Text, StyleSheet, Image } from 'react-native';
import AppHeader from '../components/AppHeader';
import { appStyle } from '../theme/style';
import { useNavigation } from '@react-navigation/native';
import { Button } from 'react-native-paper';
import { useState } from 'react';
import SelectionModal from '../components/modals/SelectionModal';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import theme from '../theme/theme';

const ResultScreen = ({ route }) => {
  const navigation = useNavigation();
  const [showSelectModal, setShowSelectModal] = useState(false);
  const { photoUri } = route.params || {};

  //Result screen have a different layout than the rest, so it should only have the header
  return (
    <View style={appStyle.container}>
      <AppHeader />
      <ScrollView contentContainerStyle={{
        backgroundColor: "#eeeeeeff", alignItems: 'center', flex: 1, justifyContent: 'center',
        borderRadius: 10, paddingBottom: 10, paddingTop: 10, paddingRight: 30, paddingLeft: 30
      }}>
        {photoUri ? (
          <Image source={{ uri: photoUri }} style={styles.image} />
        ) : (
          <Text>No image captured</Text>
        )}
        <Text style={styles.title}>Result</Text>
        <Text style={styles.result}>Name of Result</Text>
        <View style={{
          width: '100%', height: 20, backgroundColor: '#40e778ff', justifyContent: 'center',
          alignItems: 'center', borderRadius: 10
        }}>
          <Text>Percentage here</Text>
        </View>
        <Text style={styles.description}>Description of scanned pomelo images should go here</Text>

        {/*button containers for retry and home*/}
        <View style={{ width: '100%', height: 200, flexDirection: 'row', justifyContent: 'space-between' }}>
          <Button
            mode="text"
            contentStyle={{
              height: 80,
              justifyContent: "center",
            }}
            onPress={() => setShowSelectModal(!showSelectModal ? true : false)}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <MaterialIcons name="settings-backup-restore" color="#000000ff" size={26} />
              <Text style={{ color: "#000000ff", marginLeft: 8, fontSize: 20 }}>Retry</Text>
            </View>
          </Button>
          <Button
            mode="text"
            contentStyle={{
              height: 80,
              justifyContent: "center",
            }}
            onPress={() => navigation.navigate("Main")}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <MaterialIcons name="home" color="#000000ff" size={26} />
              <Text style={{ color: "#000000ff", marginLeft: 8, fontSize: 20 }}>Home</Text>
            </View>
          </Button>
        </View>

      </ScrollView>

      {showSelectModal && <SelectionModal isVisible={showSelectModal}></SelectionModal>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 18, fontWeight: 'bold', marginBottom: 10 },
  result: { fontSize: 24, fontWeight: '500', paddingBottom: 10, color: theme.colors.primary },
  image: { width: 300, height: 300, marginVertical: 10, borderRadius: 20 },
  description: { textAlign: 'justify', paddingTop: 40, paddingBottom: 40 }

});

export default ResultScreen;
