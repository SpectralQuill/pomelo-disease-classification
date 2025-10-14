import { View, Image } from "react-native";
import { Appbar, IconButton } from "react-native-paper";
import { appStyle } from "../theme/style";
import theme from "../theme/theme";
import { MaterialIcons, Foundation } from '@expo/vector-icons'

const AppHeader = () => {
    return (
        <Appbar.Header style={{ backgroundColor: theme.colors.primary }}>
            <View style={appStyle.appBar}>
                <Image
                    source={require('../assets/app-logo-white.png')}
                    style={{ width: 50, height: 50, marginRight: 8 }}
                    resizeMode="contain"
                />
                <Appbar.Content title="POMELO" titleStyle={appStyle.logo_title} />
                <IconButton icon={() => (<Foundation name="list" size={26} color={'#fff'} />)}></IconButton>
            </View>
        </Appbar.Header>
    );
}
export default AppHeader;