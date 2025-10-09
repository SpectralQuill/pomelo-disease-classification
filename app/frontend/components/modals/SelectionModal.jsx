import {View, Text, StyleSheet, Image, Modal} from 'react-native';
import { Button } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import theme from '../../theme/theme';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';

const SelectionModal = ({isVisible}) => {
    const navigation = useNavigation();

    return (
        <Modal animationType='slide' visible={isVisible}>
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
      </Modal>
    );
}

export default SelectionModal;