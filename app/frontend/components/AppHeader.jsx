import { View, Image } from "react-native";
import { Appbar } from "react-native-paper";
import { appStyle } from "../theme/style";
import theme from "../theme/theme";

const AppHeader = () => {
    return (
        <Appbar.Header style={{backgroundColor: theme.colors.primary}}>
            <View style={appStyle.appBar}>
                <Image
                    source={require('../assets/app-logo-white.png')}
                    style={{ width: 50, height: 50, marginRight: 8 }}
                    resizeMode="contain"
                />
                <Appbar.Content title = "POMELO" titleStyle={appStyle.logo_title}/>
            </View>
        </Appbar.Header>
    );
}
export default AppHeader;