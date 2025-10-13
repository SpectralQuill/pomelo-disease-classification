import React, { useState, useEffect, useRef } from 'react';
import { View, ScrollView, Text, StyleSheet, Image } from 'react-native';
import AppHeader from '../components/AppHeader';
import { appStyle } from '../theme/style';
import { useNavigation } from '@react-navigation/native';
import { Button } from 'react-native-paper';
import SelectionModal from '../components/modals/SelectionModal';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import theme from '../theme/theme';
import classificationService from '../services/classificationService';
import LoadingOverlay from '../components/LoadingOverlay';

const ResultScreen = ({ route }) => {
  const navigation = useNavigation();
  const [loading, setLoading] = useState(false);
  const [showSelectModal, setShowSelectModal] = useState(false);
  const { photoUri } = route.params || {};
  const [result, setResult] = useState(null);
  const [confidenceColor, setConfidenceColor] = useState('#6b6b6bff');


  const handleCloseModal = () => {
    setShowSelectModal(false);
  }
  useEffect(() => {
    if (backendStatus === 'connected') {
      classifyImage();
    }
  }, [backendStatus]);

  const classifyImage = async () => {
    if (!photoUri) {
      Alert.alert('No images detected, Please retry');
      return;
    }

    if (backendStatus !== 'connected') {
      Alert.alert('Backend Unavailable', 'Please ensure the backend server is running and connected.');
      return;
    }

    setLoading(true);
    try {
      console.log('Starting classification...');
      const classificationResult = await classificationService.classifyImage(selectedImage);
      setResult(classificationResult);
      handlePredictionColor();
      console.log('Classification result:', classificationResult);
    } catch (error) {
      Alert.alert('Classification Error', error.message);
      console.error('Classification error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictionColor = () => {
    if ((result.confidence * 100).toFixed(1) <= 50) { setConfidenceColor('#FF4C4C') }
    if ((result.confidence * 100).toFixed(1) > 50 && (result.confidence * 100).toFixed(1) <= 70) setConfidenceColor('#FFA500');
    if ((result.confidence * 100).toFixed(1) > 70 && (result.confidence * 100).toFixed(1) <= 85) setConfidenceColor('#FFD700');
    if ((result.confidence * 100).toFixed(1) > 85 && (result.confidence * 100).toFixed(1) <= 95) setConfidenceColor('#9ACD32');
    if ((result.confidence * 100).toFixed(1) > 95) setConfidenceColor('#00C853');
    else setConfidenceColor('#858585ff');
    return;
  }

  //Result screen have a different layout than the rest, so it should only have the header
  return (
    <View style={appStyle.container}>
      {loading && <LoadingOverlay />}
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
        <Text style={styles.result}>{result.predicted_class}</Text>
        <View style={{
          width: '100%', height: 20, backgroundColor: confidenceColor, justifyContent: 'center',
          alignItems: 'center', borderRadius: 10
        }}>
          <Text>Confidence: {(result.confidence * 100).toFixed(1)}%</Text>
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

      {showSelectModal && <SelectionModal isVisible={showSelectModal} onClose={handleCloseModal}></SelectionModal>}
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
