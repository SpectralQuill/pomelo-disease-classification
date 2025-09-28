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