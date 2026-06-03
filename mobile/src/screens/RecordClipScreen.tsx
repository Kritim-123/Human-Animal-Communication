import { Audio } from 'expo-av';
import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Button, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { Dog, DogStats, getStats, listDogs, uploadClip } from '../services/api';

const labels = ['outside_bathroom', 'food_water', 'play', 'attention', 'stress_discomfort', 'unknown'];
const locations = ['door', 'kitchen', 'couch', 'crate', 'outside', 'unknown'];
const situations = ['before_walk', 'before_food', 'owner_leaving', 'stranger_nearby', 'toy_visible', 'unknown'];

export default function RecordClipScreen() {
  const [dogs, setDogs] = useState<Dog[]>([]);
  const [selectedDogId, setSelectedDogId] = useState<number | null>(null);
  const [stats, setStats] = useState<DogStats | null>(null);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recordingUri, setRecordingUri] = useState<string | null>(null);
  const [ownerLabel, setOwnerLabel] = useState('unknown');
  const [outcomeLabel, setOutcomeLabel] = useState('unknown');
  const [location, setLocation] = useState('unknown');
  const [situation, setSituation] = useState('unknown');
  const [notes, setNotes] = useState('');
  const [loadingDogs, setLoadingDogs] = useState(false);
  const [uploading, setUploading] = useState(false);

  const selectedDog = useMemo(() => dogs.find((dog) => dog.id === selectedDogId) ?? null, [dogs, selectedDogId]);

  useEffect(() => {
    refreshDogs();
  }, []);

  useEffect(() => {
    if (selectedDogId) {
      refreshStats(selectedDogId);
    } else {
      setStats(null);
    }
  }, [selectedDogId]);

  async function refreshDogs() {
    setLoadingDogs(true);
    try {
      const nextDogs = await listDogs();
      setDogs(nextDogs);
      setSelectedDogId((current) => current ?? nextDogs[0]?.id ?? null);
    } catch (error) {
      Alert.alert('Could not load dogs', String(error));
    } finally {
      setLoadingDogs(false);
    }
  }

  async function refreshStats(dogId: number) {
    try {
      setStats(await getStats(dogId));
    } catch {
      setStats(null);
    }
  }

  async function startRecording() {
    if (!selectedDogId) {
      Alert.alert('Create a dog first', 'Add a dog profile before recording clips.');
      return;
    }

    const permission = await Audio.requestPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Microphone permission needed', 'DogBridge needs microphone access to record dog audio clips.');
      return;
    }

    try {
      setRecordingUri(null);
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const result = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      setRecording(result.recording);
    } catch (error) {
      Alert.alert('Could not start recording', String(error));
    }
  }

  async function stopRecording() {
    if (!recording) {
      return;
    }

    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      setRecordingUri(uri);
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
    } catch (error) {
      setRecording(null);
      Alert.alert('Could not stop recording', String(error));
    }
  }

  async function submitClip() {
    if (!selectedDogId || !recordingUri) {
      Alert.alert('Missing recording', 'Record a clip before submitting.');
      return;
    }

    setUploading(true);
    try {
      const clip = await uploadClip({
        dogId: selectedDogId,
        audioUri: recordingUri,
        locationContext: location,
        situationContext: situation,
        ownerLabel,
        outcomeLabel,
        notes: notes.trim() || undefined,
      });
      setRecordingUri(null);
      setNotes('');
      await refreshStats(selectedDogId);
      Alert.alert('Clip saved', `Saved clip #${clip.id} for ${selectedDog?.name ?? 'your dog'}.`);
    } catch (error) {
      Alert.alert('Could not upload clip', String(error));
    } finally {
      setUploading(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.headerRow}>
        <Text style={styles.sectionTitle}>Dog</Text>
        <Button title="Refresh" onPress={refreshDogs} />
      </View>
      {loadingDogs ? <ActivityIndicator /> : null}
      {dogs.length === 0 ? (
        <Text style={styles.muted}>No dogs yet. Create a dog profile first, then come back here to record clips.</Text>
      ) : (
        <Segmented values={dogs.map((dog) => `${dog.id}: ${dog.name}`)} selected={`${selectedDog?.id}: ${selectedDog?.name}`} onSelect={(value) => setSelectedDogId(Number(value.split(':')[0]))} />
      )}

      <Text style={styles.sectionTitle}>Record</Text>
      <Text style={styles.muted}>Capture one clear vocal event, ideally 3-10 seconds. Label what happened, not what you hope the model will learn.</Text>
      <View style={styles.actions}>
        <Button title={recording ? 'Stop recording' : 'Start recording'} onPress={recording ? stopRecording : startRecording} />
        {recordingUri ? <Text style={styles.ready}>Recording ready to upload.</Text> : null}
      </View>

      <Text style={styles.sectionTitle}>Owner label</Text>
      <Segmented values={labels} selected={ownerLabel} onSelect={setOwnerLabel} />

      <Text style={styles.sectionTitle}>Outcome label</Text>
      <Segmented values={labels} selected={outcomeLabel} onSelect={setOutcomeLabel} />

      <Text style={styles.sectionTitle}>Location context</Text>
      <Segmented values={locations} selected={location} onSelect={setLocation} />

      <Text style={styles.sectionTitle}>Situation context</Text>
      <Segmented values={situations} selected={situation} onSelect={setSituation} />

      <Text style={styles.label}>Notes</Text>
      <TextInput style={[styles.input, styles.notes]} value={notes} onChangeText={setNotes} multiline placeholder="What happened before and after the sound?" />

      <Button title={uploading ? 'Uploading...' : 'Submit clip'} onPress={submitClip} disabled={uploading || !recordingUri || !selectedDogId} />

      {stats ? (
        <View style={styles.progress}>
          <Text style={styles.sectionTitle}>Dataset progress</Text>
          <Text>Total clips: {stats.clip_count}</Text>
          <Text>Confirmed clips: {stats.confirmed_count}</Text>
          {labels.map((label) => (
            <Text key={label}>
              {label}: {stats.label_distribution[label] ?? 0}
            </Text>
          ))}
        </View>
      ) : null}
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
  headerRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  sectionTitle: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '700',
  },
  label: {
    fontWeight: '600',
  },
  muted: {
    color: '#555',
    lineHeight: 20,
  },
  ready: {
    color: '#176b39',
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
  progress: {
    borderTopWidth: 1,
    borderTopColor: '#d8d8d8',
    gap: 6,
    marginTop: 12,
    paddingTop: 12,
  },
});
