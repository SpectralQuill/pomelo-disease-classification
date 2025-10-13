import { View, Image } from "react-native";

const LoadingOverlay = () => {
    return (
        <View style={{ flex: 1, flexDirection: 'column', justifyContent: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
            <Image source={require('../assets/app-logo.png')}> </Image>
            <Text>Processing... Please wait</Text>
        </View>
    );
}

export default LoadingOverlay;