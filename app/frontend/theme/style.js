import { StyleSheet } from "react-native";
import { theme } from './theme';

export const appStyle = StyleSheet.create({
    logo_large: {
        width: "300px",
        height: "300px"
    },
    container: {
        flex: 1,
        backgroundColor: 'white',
    },
    camera_container: {
        flex: 1,
        justifyContent: 'space-around'
    },
    button_container: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        position: 'absolute',
        bottom: 0,
        width: '110%',
        height: 120,
        alignSelf: 'center',
        borderRadius: 20,
        backgroundColor: 'rgba(0,0,0,0.8)',
        justifyContent: 'center',
        alignItems: 'center',
        paddingVertical: 10,
    },
    appBar: {
        flexDirection: 'row',
        alignItems: 'center',
        flex: '1',
        padding: 20
    },
    logo_title: {
        fontSize: 24,
        fontWeight: 'bold',
        color: "#FFFFFF"
    },
    centered: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    screen: {
        flex: 1,
        padding: 30,
    }

});