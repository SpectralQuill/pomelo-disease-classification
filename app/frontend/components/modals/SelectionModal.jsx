import { View, Text, StyleSheet, Image, Modal } from 'react-native';
import { Button, IconButton } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import theme from '../../theme/theme';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';

const SelectionModal = ({ isVisible, onClose }) => {
    const navigation = useNavigation();

    return (
        <Modal
            animationType='slide'
            transparent={true}
            visible={isVisible}
        >
            <View style={styles.modalBackground}>
                <View style={styles.modalContainer}>
                    <Text style={{ paddingBottom: 20, textAlign: 'left', alignSelf: 'flex-start', fontSize: 18 }}>Choose an action</Text>

                    <Button
                        mode="contained"
                        style={styles.button}
                        contentStyle={styles.buttonContent}
                        onPress={() => navigation.navigate("Scanner")}
                    >
                        <View style={styles.row}>
                            <MaterialIcons name="photo-camera" color="#fff" size={26} />
                            <Text style={styles.buttonText}>Scan from Camera</Text>
                        </View>
                    </Button>

                    <Button
                        mode="contained"
                        style={styles.button}
                        contentStyle={styles.buttonContent}
                        onPress={() => navigation.navigate("Picker")}
                    >
                        <View style={styles.row}>
                            <MaterialIcons name="photo-library" color="#fff" size={26} />
                            <Text style={styles.buttonText}>Open from Gallery</Text>
                        </View>
                    </Button>

                    <Button
                        mode="contained"
                        style={styles.button}
                        contentStyle={styles.buttonContent}
                        onPress={onClose}
                    >
                        <View style={styles.row}>
                            <MaterialIcons name="close" color="#fff" size={26} />
                            <Text style={styles.buttonText}>Cancel</Text>
                        </View>
                    </Button>
                </View>
            </View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    modalBackground: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    modalContainer: {
        backgroundColor: '#eeeeee',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 10,
        padding: 30,
        width: '85%',
    },
    button: {
        width: '100%',
        backgroundColor: theme.colors.primary,
        marginBottom: 10,
        marginTop: 10,
    },
    buttonContent: {
        height: 60,
        justifyContent: 'center',
    },
    row: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    buttonText: {
        color: '#fff',
        marginLeft: 8,
        fontSize: 20,
    },
});

export default SelectionModal;