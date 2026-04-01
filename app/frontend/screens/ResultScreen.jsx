import React, { useState, useEffect, useRef } from 'react';
import { View, ScrollView, Text, StyleSheet, Image, ImageBackground, Alert, Linking } from 'react-native';
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
import diseases from '../assets/json/diseases.json';
import SourcesModal from '../components/modals/SourcesModal';

const ResultScreen = ({ route }) => {
  const { isConnected, isChecking, refreshConnection } = useServerConnection();
  const navigation = useNavigation();
  const [loading, setLoading] = useState(true);
  const [showSelectModal, setShowSelectModal] = useState(false);
  const { photoUri } = route.params || {};
  const [result, setResult] = useState(null);
  const [confidenceColor, setConfidenceColor] = useState('#6b6b6bff');
  const [diseaseDetail, setDiseaseDetail] = useState(null);
  const [showSymptoms, setShowSymptoms] = useState(false);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [showSourcesModal, setShowSourcesModal] = useState(false);


  useEffect(() => {
    if (isConnected) {
      classifyImage();
    }
  }, [isConnected]);

  useEffect(() => {
    if (result) {
      handlePredictionColor(result?.confidence ?? 0);
      handleDiseaseDetails(result?.predicted_class);
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

  const handleCloseModal = () => {
    setShowSelectModal(false);
  }

  const handleShowSymptoms = () => {
    setShowSymptoms(!showSymptoms ? true : false);
  }

  const handleShowRecommendations = () => {
    setShowRecommendations(!showRecommendations ? true : false);
  }

  const handleCloseSources = () => {
    setShowSourcesModal(false);
  }

  //function to display the disease's details
  const handleDiseaseDetails = (diseaseType) => {
    switch (diseaseType) {
      case "anthracnose":
        setDiseaseDetail(diseases.pomelo_diseases[0]);
        break;
      case "blackspot":
        setDiseaseDetail(diseases.pomelo_diseases[1]);
        break;
      case "borer":
        setDiseaseDetail(diseases.pomelo_diseases[2]);
        break;
      case "melanose":
        setDiseaseDetail(diseases.pomelo_diseases[3]);
      case "mites":
        setDiseaseDetail(diseases.pomelo_diseases[4]);
        break;
      case "healthy":
        setDiseaseDetail(diseases.pomelo_diseases[5]);
        break;
      case "others":
        setDiseaseDetail(diseases.pomelo_diseases[6]);
        break;
      default:
        setDiseaseDetail(null);
        break;
    }
  }
  return (
    <View style={appStyle.container}>
      <AppHeader />
      {loading && <LoadingOverlay />}
      {result &&
        <ScrollView contentContainerStyle={{
          alignItems: 'center', justifyContent: 'center',
          paddingBottom: 10, paddingTop: 10, paddingRight: 30, paddingLeft: 30,
        }}>
          <View style={{ marginTop: 10 }}>
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

          {/* <View style={{
            width: '100%', height: 35, backgroundColor: confidenceColor, justifyContent: 'center',
            alignItems: 'center', borderRadius: 20
          }}>
            <Text style={styles.percentage}>Confidence: {(result.confidence * 100).toFixed(1)}%</Text>
          </View> */}

          {/*disease details section*/}
          {diseaseDetail && (
            <>
              <Text style={styles.title}>Result</Text>
              <Text style={styles.result}>{diseaseDetail?.name}</Text>
              <Text style={styles.descriptionText}>{diseaseDetail?.description}</Text>

              {/*Since healthy and others doesn't have a symptoms, why not just hide this if it was the case*/}
              {(diseaseDetail.name != "Healthy" || diseaseDetail.name != "Others") && (
                <View style={{ width: '100%', marginBottom: 30 }}>
                  <Button
                    mode="contained"
                    onPress={handleShowSymptoms}
                    style={styles.headerButton}
                    contentStyle={{ justifyContent: 'flex-start' }}
                    labelStyle={{ fontSize: 18, fontWeight: 'bold', color: '#333' }}
                    icon={showSymptoms ? 'chevron-up' : 'chevron-down'}

                  >
                    <Text style={styles.headerText}>Symptoms</Text>
                  </Button>

                  {showSymptoms && (
                    <View style={{ marginLeft: 10, marginTop: 10 }}>
                      {diseaseDetail.symptoms?.map((symptom, index) => (
                        <Text key={index} style={styles.sectionsText}>
                          • {symptom}
                        </Text>
                      ))}
                    </View>
                  )}
                </View>
              )}

              <View style={{ width: '100%' }}>
                <Button
                  mode="text"
                  onPress={handleShowRecommendations}
                  style={styles.headerButton}
                  contentStyle={{ justifyContent: 'flex-start' }}
                  labelStyle={{ fontSize: 18, fontWeight: 'bold', color: '#333' }}
                  icon={showRecommendations ? 'chevron-up' : 'chevron-down'}
                >
                  <Text style={styles.headerText}>Recommendations</Text>
                </Button>

                {showRecommendations && (
                  <View style={{ marginLeft: 10, marginTop: 10 }}>
                    {diseaseDetail.recommendations?.map((recommendation, index) => (
                      <Text key={index} style={styles.sectionsText}>
                        • {recommendation}
                      </Text>
                    ))}
                  </View>
                )}
              </View>

              <Button
                mode="contained"
                style={styles.sourceButton}
                contentStyle={{ height: 60 }} // only controls height
                onPress={() => setShowSourcesModal(true)}
              >
                <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
                  <Text style={{ color: '#fff', marginRight: 8, fontSize: 16, textAlignVertical: 'center' }}>
                    Learn more about the Disease
                  </Text>
                  <MaterialIcons name="arrow-circle-right" color="#fff" size={24} />
                </View>
              </Button>

            </>
          )}

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
      {showSourcesModal && <SourcesModal isVisible={showSourcesModal} onClose={handleCloseSources} sources={diseaseDetail?.source}></SourcesModal>}
    </View >
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 18, fontWeight: 'bold', marginBottom: 10 },
  result: { fontSize: 26, fontWeight: '900', paddingBottom: 10, color: theme.colors.primary },
  image: { width: 300, height: 300, marginVertical: 10, borderRadius: 20 },
  descriptionText: { textAlign: 'justify', paddingBottom: 40, paddingTop: 20, fontWeight: '400', fontSize: 15 },
  sectionsText: { textAlign: 'justify', paddingBottom: 3 },
  percentage: { color: 'white', fontWeight: '700' },

  headerButton: {
    width: '100%',
    backgroundColor: "#D3E8CF",
    alignSelf: 'flex',
  },

  sourceButton: {
    paddingVertical: 5,
    marginTop: 40,
    marginBottom: 10,
    width: '100%',
    backgroundColor: theme.colors.primary,
  },
  headerText: {
    textAlign: 'left',
    alignSelf: 'flex-start',
    fontWeight: '900',
    color: theme.colors.primary,
  },
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
