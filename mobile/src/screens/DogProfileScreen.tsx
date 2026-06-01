import React, { useState } from 'react';
import { Alert, Button, StyleSheet, Text, TextInput, View } from 'react-native';

import { createDog } from '../services/api';

export default function DogProfileScreen() {
  const [name, setName] = useState('');
  const [breed, setBreed] = useState('');
  const [notes, setNotes] = useState('');

  async function handleSubmit() {
    if (!name.trim()) {
      Alert.alert('Name required', 'Add your dog name first.');
      return;
    }
    try {
      await createDog({ name, breed: breed || undefined, notes: notes || undefined });
      Alert.alert('Saved', 'Dog profile created.');
      setName('');
      setBreed('');
      setNotes('');
    } catch (error) {
      Alert.alert('Could not save', String(error));
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Dog name</Text>
      <TextInput style={styles.input} value={name} onChangeText={setName} placeholder="Milo" />
      <Text style={styles.label}>Breed</Text>
      <TextInput style={styles.input} value={breed} onChangeText={setBreed} placeholder="Optional" />
      <Text style={styles.label}>Notes</Text>
      <TextInput style={[styles.input, styles.notes]} value={notes} onChangeText={setNotes} multiline />
      <Button title="Save profile" onPress={handleSubmit} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 10,
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
    minHeight: 96,
    textAlignVertical: 'top',
  },
});

