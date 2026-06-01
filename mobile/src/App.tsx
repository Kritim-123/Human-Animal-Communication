import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';

import DogProfileScreen from './screens/DogProfileScreen';
import HomeScreen from './screens/HomeScreen';
import PredictionScreen from './screens/PredictionScreen';
import RecordClipScreen from './screens/RecordClipScreen';

export type RootStackParamList = {
  Home: undefined;
  DogProfile: undefined;
  RecordClip: undefined;
  Prediction: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="dark" />
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} options={{ title: 'DogBridge' }} />
        <Stack.Screen name="DogProfile" component={DogProfileScreen} options={{ title: 'Dog Profile' }} />
        <Stack.Screen name="RecordClip" component={RecordClipScreen} options={{ title: 'Record Clip' }} />
        <Stack.Screen name="Prediction" component={PredictionScreen} options={{ title: 'Prediction' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

