import React from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { RootStackParamList } from '../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>DogBridge</Text>
      <Text style={styles.copy}>
        Collect real dog audio, add context, and review likely intent predictions with confidence.
      </Text>
      <View style={styles.actions}>
        <Button title="Create dog profile" onPress={() => navigation.navigate('DogProfile')} />
        <Button title="Record dog sound" onPress={() => navigation.navigate('RecordClip')} />
        <Button title="View prediction" onPress={() => navigation.navigate('Prediction')} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    justifyContent: 'center',
    gap: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
  },
  copy: {
    fontSize: 16,
    lineHeight: 24,
  },
  actions: {
    gap: 12,
  },
});

