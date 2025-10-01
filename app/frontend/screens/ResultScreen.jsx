import { View, Text, StyleSheet, Image } from 'react-native';
import AppHeader from '../components/AppHeader';
import { appStyle } from '../theme/style';

const ResultScreen = ({ route }) => {
  const { photoUri } = route.params || {};

  //Result screen have a different layout than the rest, so it should only have the header
  return (
    <View style={appStyle.container}>
      <AppHeader/>
      {photoUri ? (
        <Image source={{ uri: photoUri }} style={styles.image} />
      ) : (
        <Text>No image captured</Text>
      )}
      <Text style={styles.title}>Result</Text>
      
      <Text>Description of scanned pomelo goes here</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 10 },
  image: { width: 250, height: 250, marginVertical: 10 },
});

export default ResultScreen;
