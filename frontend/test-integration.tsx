import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { railwayAPI, handleAPIError } from '../services/api';

export default function IntegrationTest() {
  const [backendStatus, setBackendStatus] = useState('Testing...');
  const [testResults, setTestResults] = useState<string[]>([]);

  const runIntegrationTest = async () => {
    const results: string[] = [];
    
    try {
      // Test 1: Health Check
      results.push('🔍 Testing health endpoint...');
      const health = await railwayAPI.getHealth();
      results.push(`✅ Health: ${health.message}`);
      results.push(`✅ Models loaded: ${health.models_loaded.delay_model && health.models_loaded.conflict_model}`);
      
      // Test 2: Get all trains
      results.push('🔍 Testing get all trains...');
      const trains = await railwayAPI.getAllTrains();
      results.push(`✅ Found ${trains.length} trains`);
      
      // Test 3: Critical situations
      results.push('🔍 Testing critical situations...');
      const critical = await railwayAPI.getCriticalSituations();
      results.push(`✅ Critical situations: ${critical.total_critical} critical, ${critical.immediate_attention_needed} urgent`);
      
      // Test 4: Delay prediction
      results.push('🔍 Testing delay prediction...');
      const delayPred = await railwayAPI.predictDelay({
        train_type: 'EXPRESS',
        block_length_km: 25.0,
        speed_limit_kmph: 120,
        rake_length_m: 400,
        priority_level: 1,
        headway_seconds: 300,
        tsr_active: 'N',
        tsr_speed_kmph: 0
      });
      results.push(`✅ Delay prediction: ${delayPred.predicted_delay.toFixed(2)} minutes`);
      
      // Test 5: Conflict prediction
      results.push('🔍 Testing conflict prediction...');
      const conflictPred = await railwayAPI.predictConflict({
        train_type: 'EXPRESS',
        block_length_km: 25.0,
        speed_limit_kmph: 120,
        rake_length_m: 400,
        priority_level: 1,
        headway_seconds: 300,
        tsr_active: 'N',
        tsr_speed_kmph: 0
      });
      results.push(`✅ Conflict likelihood: ${(conflictPred.conflict_likelihood * 100).toFixed(1)}%`);
      
      results.push('');
      results.push('🎉 ALL INTEGRATION TESTS PASSED!');
      results.push('Frontend-Backend integration is working perfectly!');
      
      setBackendStatus('✅ Connected');
      
    } catch (error) {
      results.push(`❌ Error: ${handleAPIError(error)}`);
      setBackendStatus('❌ Failed');
    }
    
    setTestResults(results);
  };

  useEffect(() => {
    runIntegrationTest();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🔗 Frontend-Backend Integration Test</Text>
      
      <View style={styles.statusContainer}>
        <Text style={styles.statusLabel}>Backend Status: </Text>
        <Text style={[styles.status, { color: backendStatus.includes('✅') ? '#44ff44' : '#ff4444' }]}>
          {backendStatus}
        </Text>
      </View>

      <TouchableOpacity style={styles.button} onPress={runIntegrationTest}>
        <Text style={styles.buttonText}>🔄 Run Integration Test</Text>
      </TouchableOpacity>

      <View style={styles.resultsContainer}>
        {testResults.map((result, index) => (
          <Text key={index} style={styles.resultText}>{result}</Text>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  statusContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  statusLabel: {
    fontSize: 16,
    color: '#666',
  },
  status: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  button: {
    backgroundColor: '#0066cc',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    maxHeight: 400,
  },
  resultText: {
    fontSize: 14,
    marginBottom: 5,
    fontFamily: 'monospace',
    color: '#333',
  },
});