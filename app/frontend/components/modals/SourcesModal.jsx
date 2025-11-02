import React from 'react';
import { View, Text, StyleSheet, Modal, Linking } from 'react-native';
import { Button } from 'react-native-paper';
import theme from '../../theme/theme';

const SourcesModal = ({ isVisible, onClose, sources = [] }) => {
    return (
        <Modal
            visible={isVisible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <View style={styles.modalOverlay}>
                <View style={styles.modalBox}>
                    <Text style={styles.modalTitle}>Sources</Text>

                    {sources && sources.length > 0 ? (
                        sources.map((src, index) => (
                            <Text
                                key={index}
                                style={styles.modalLink}
                                onPress={() => Linking.openURL(src)}
                            >
                                • {src}
                            </Text>
                        ))
                    ) : (
                        <Text style={{ textAlign: 'center', color: '#777' }}>
                            No sources available
                        </Text>
                    )}

                    <Button
                        mode="contained"
                        style={{ marginTop: 20, backgroundColor: theme.colors.primary }}
                        onPress={onClose}
                    >
                        Close
                    </Button>
                </View>
            </View>
        </Modal>
    );
};

export default SourcesModal;

const styles = StyleSheet.create({
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.6)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    modalBox: {
        backgroundColor: 'white',
        borderRadius: 10,
        width: '85%',
        padding: 20,
        elevation: 5,
    },
    modalTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 10,
        textAlign: 'center',
    },
    modalLink: {
        color: theme.colors.primary,
        marginBottom: 6,
        textDecorationLine: 'underline',
    },
});
