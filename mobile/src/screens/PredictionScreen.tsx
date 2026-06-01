import React, { useState } from 'react';
import { Alert, Button, StyleSheet, Text, TextInput, View } from 'react-native';

import { confirmPrediction, predictClip } from '../services/api';

export default function PredictionScreen() {
  const [clipId, setClipId] = useState('1');
  const [prediction, setPrediction] = useState<{ predicted_label: string; confidence: number; message?: string } | null>(null);

  async function handlePredict() {
    try {
      const result = await predictClip(Number(clipId));
      setPrediction(result);
    } catch (error) {
      Alert.alert('Prediction unavailable', String(error));
    }
  }

  async function handleConfirm(confirmedCorrect: boolean) {
    try {
      await confirmPrediction(Number(clipId), { confirmed_correct: confirmedCorrect });
      Alert.alert('Saved', 'Your feedback was stored.');
    } catch (error) {
      Alert.alert('Could not confirm', String(error));
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Clip ID</Text>
      <TextInput style={styles.input} value={clipId} onChangeText={setClipId} keyboardType="number-pad" />
      <Button title="Predict likely intent" onPress={handlePredict} />
      {prediction && (
        <View style={styles.result}>
          <Text style={styles.title}>Possible intent: {prediction.predicted_label}</Text>
          <Text>Confidence: {Math.round(prediction.confidence * 100)}%</Text>
          {prediction.message ? <Text style={styles.warning}>{prediction.message}</Text> : null}
          <View style={styles.actions}>
            <Button title="Correct" onPress={() => handleConfirm(true)} />
            <Button title="Incorrect" onPress={() => handleConfirm(false)} />
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 12,
  },
  label: {
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderColor: '#c8c8c8',
    borderRadius: 8,
    padding: 12,
  },
  result: {
    marginTop: 12,
    gap: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
  },
  warning: {
    color: '#8a4b00',
    lineHeight: 20,
  },
  actions: {
    gap: 12,
  },
});

