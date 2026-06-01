import React, { useState } from 'react';
import { Alert, Button, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

const labels = ['outside_bathroom', 'food_water', 'play', 'attention', 'stress_discomfort', 'unknown'];
const locations = ['door', 'kitchen', 'couch', 'crate', 'outside', 'unknown'];
const situations = ['before_walk', 'before_food', 'owner_leaving', 'stranger_nearby', 'toy_visible', 'unknown'];

export default function RecordClipScreen() {
  const [dogId, setDogId] = useState('1');
  const [ownerLabel, setOwnerLabel] = useState('unknown');
  const [location, setLocation] = useState('unknown');
  const [situation, setSituation] = useState('unknown');
  const [notes, setNotes] = useState('');

  function handleRecord() {
    Alert.alert('Recording placeholder', 'Audio recording and upload wiring comes next. This screen captures the MVP context fields.');
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.sectionTitle}>Clip</Text>
      <Text style={styles.label}>Dog ID</Text>
      <TextInput style={styles.input} value={dogId} onChangeText={setDogId} keyboardType="number-pad" />

      <Text style={styles.sectionTitle}>Owner label</Text>
      <Segmented values={labels} selected={ownerLabel} onSelect={setOwnerLabel} />

      <Text style={styles.sectionTitle}>Location context</Text>
      <Segmented values={locations} selected={location} onSelect={setLocation} />

      <Text style={styles.sectionTitle}>Situation context</Text>
      <Segmented values={situations} selected={situation} onSelect={setSituation} />

      <Text style={styles.label}>Notes</Text>
      <TextInput style={[styles.input, styles.notes]} value={notes} onChangeText={setNotes} multiline />

      <View style={styles.actions}>
        <Button title="Record audio" onPress={handleRecord} />
        <Button title="Submit clip placeholder" onPress={handleRecord} />
      </View>
    </ScrollView>
  );
}

function Segmented({ values, selected, onSelect }: { values: string[]; selected: string; onSelect: (value: string) => void }) {
  return (
    <View style={styles.segmentWrap}>
      {values.map((value) => (
        <Text key={value} style={[styles.segment, selected === value && styles.segmentSelected]} onPress={() => onSelect(value)}>
          {value}
        </Text>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    gap: 12,
  },
  sectionTitle: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '700',
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
  notes: {
    minHeight: 92,
    textAlignVertical: 'top',
  },
  segmentWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  segment: {
    borderWidth: 1,
    borderColor: '#b8b8b8',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
  },
  segmentSelected: {
    backgroundColor: '#163d5c',
    borderColor: '#163d5c',
    color: '#ffffff',
  },
  actions: {
    gap: 12,
    marginTop: 8,
  },
});

