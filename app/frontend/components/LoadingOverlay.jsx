import { View, Image, Text, StyleSheet } from "react-native";

const LoadingOverlay = () => {
    return (
        <View style={styles.overlay}>
            <View style={styles.content}>
                <Image source={require('../assets/app-logo.png')} style={styles.logo} />
                <Text style={styles.text}>Processing... Please wait</Text>
            </View>

        </View>
    );
}

const styles = StyleSheet.create({
    overlay: {
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 999,
    },
    content: {
        alignItems: 'center'
    },
    logo: {
        width: 250,
        height: 250,
        marginBottom: 16,
    },
    text: {
        color: "#fff",
        fontSize: 18,
        fontWeight: "500",
    },
});

export default LoadingOverlay;

