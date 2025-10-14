import React, { useState, useEffect, useRef } from 'react';
import { View, ScrollView, Text, StyleSheet, Image, ImageBackground, Alert } from 'react-native';
import AppHeader from '../components/AppHeader';
import { appStyle } from '../theme/style';
import { useNavigation } from '@react-navigation/native';
import { Button } from 'react-native-paper';
import SelectionModal from '../components/modals/SelectionModal';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import theme from '../theme/theme';
import classificationService from '../services/classificationService';
import LoadingOverlay from '../components/LoadingOverlay';
import { useServerConnection } from '../context/ServerConnectionContext';


const ResultScreen = ({ route }) => {
  const { isConnected, isChecking, refreshConnection } = useServerConnection();
  console.log('Type:', typeof classificationService);
  console.log('Keys:', Object.keys(classificationService));

  const navigation = useNavigation();
  const [loading, setLoading] = useState(true);
  const [showSelectModal, setShowSelectModal] = useState(false);
  const { photoUri } = route.params || {};
  const [result, setResult] = useState(null);
  const [confidenceColor, setConfidenceColor] = useState('#6b6b6bff');


  const handleCloseModal = () => {
    setShowSelectModal(false);
  }
  useEffect(() => {
    if (isConnected) {
      classifyImage();
    }
  }, [isConnected]);

  useEffect(() => {
    if (result) {
      handlePredictionColor();
    }
  }, [result])

  const classifyImage = async () => {
    if (!photoUri) {
      Alert.alert('No images detected, Please retry');
      return;
    }

    if (!isConnected) {
      Alert.alert('Server Unavailable', 'Please ensure the server is running and connected.');
      return;
    }
    setLoading(true);
    try {
      console.log('Starting classification...');
      const classificationResult = await classificationService.classifyImage(photoUri);
      setResult(classificationResult);
      console.log('Classification result:', classificationResult);
    } catch (error) {
      Alert.alert('Classification Error', error.message);
      console.error('Classification error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictionColor = (confidence) => {
    const percent = (confidence * 100).toFixed(1);
    if (percent <= 50) return setConfidenceColor('#FF4C4C');
    if (percent <= 70) return setConfidenceColor('#FFA500');
    if (percent <= 85) return setConfidenceColor('#FFD700');
    if (percent <= 95) return setConfidenceColor('#9ACD32');
    if (percent > 95) return setConfidenceColor('#00C853');
    setConfidenceColor('#858585ff');
  };

  //Result screen have a different layout than the rest, so it should only have the header

  return (
    <View style={appStyle.container}>
      {loading && <LoadingOverlay />}
      <AppHeader />
      {result &&
        <ScrollView contentContainerStyle={{
          backgroundColor: "#eeeeeeff", alignItems: 'center', flex: 1, justifyContent: 'center',
          borderRadius: 10, paddingBottom: 10, paddingTop: 10, paddingRight: 30, paddingLeft: 30
        }}>
          <View>
            <View style={{ alignItems: 'center' }}>
              {photoUri && photoUri !== '' ? (
                <ImageBackground
                  source={{ uri: photoUri }}
                  style={styles.image}
                  imageStyle={{ borderRadius: 20 }}
                />
              ) : (
                <View style={[styles.image, { justifyContent: 'center', alignItems: 'center' }]}>
                  <Text style={styles.placeholderText}>No image selected</Text>
                  <Text style={styles.placeholderSubtext}>Choose or take a photo of pomelo leaves</Text>
                </View>
              )}
            </View>
          </View>
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

        </ScrollView>}


      {showSelectModal && <SelectionModal isVisible={showSelectModal} onClose={handleCloseModal}></SelectionModal>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 18, fontWeight: 'bold', marginBottom: 10 },
  result: { fontSize: 24, fontWeight: '500', paddingBottom: 10, color: theme.colors.primary },
  image: { width: 300, height: 300, marginVertical: 10, borderRadius: 20 },
  description: { textAlign: 'justify', paddingTop: 40, paddingBottom: 40 },
  placeholderText: {
    color: '#6c757d',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  placeholderSubtext: {
    color: '#adb5bd',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 20,
  },

});

export default ResultScreen;
