import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';

const ResultScreen = ({ route }) => {
  const { photoUri } = route.params || {};

  //Next thing to do: there should be a different view for a failed capture and a successful capture
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Result</Text>
      {photoUri ? (
        <Image source={{ uri: photoUri }} style={styles.image} />
      ) : (
        <Text>No image captured</Text>
      )}
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
