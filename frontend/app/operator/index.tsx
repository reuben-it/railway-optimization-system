import { useRouter } from "expo-router";
import React, { useEffect, useState } from 'react';
import {
    ActivityIndicator,
    Alert,
    Modal,
    RefreshControl,
    ScrollView,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View
} from "react-native";
import { CriticalSituationsResponse, handleAPIError, HealthResponse, OperatorDecision, railwayAPI, TrainInfo } from '../../services/api';

// Utility function to safely format numbers
const safeToFixed = (value: number | null | undefined, decimals: number = 1, suffix: string = ''): string => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A';
  }
  return value.toFixed(decimals) + suffix;
};

const safePercentage = (value: number | null | undefined): string => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A%';
  }
  return (value * 100).toFixed(1) + '%';
};

export default function OperatorHome() {
  const router = useRouter();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [trains, setTrains] = useState<TrainInfo[]>([]);
  const [criticalSituations, setCriticalSituations] = useState<CriticalSituationsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showAddTrain, setShowAddTrain] = useState(false);
  const [showDecisionModal, setShowDecisionModal] = useState(false);
  const [selectedTrain, setSelectedTrain] = useState<TrainInfo | null>(null);
  const [decision, setDecision] = useState({
    decision: 'allow_delay',
    delay_minutes: 0,
    new_speed_limit: 120,
    reason: '',
    priority: 'normal'
  });
  const [newTrain, setNewTrain] = useState({
    train_id: '',
    train_type: 'EXPRESS',
    current_section: '',
    from_station: '',
    to_station: '',
    scheduled_arrival: new Date().toISOString(),
    block_length_km: 25.0,
    speed_limit_kmph: 120,
    rake_length_m: 400,
    priority_level: 1,
    headway_seconds: 300
  });

  const loadData = async () => {
    try {
      const [healthData, trainsData, criticalData] = await Promise.all([
        railwayAPI.getHealth(),
        railwayAPI.getAllTrains(),
        railwayAPI.getCriticalSituations()
      ]);
      
      setHealth(healthData);
      // Ensure trainsData is an array
      setTrains(Array.isArray(trainsData) ? trainsData : []);
      setCriticalSituations(criticalData);
    } catch (error) {
      console.error('Error loading data:', error);
      Alert.alert('Error', handleAPIError(error));
      // Set default values on error
      setTrains([]);
      setCriticalSituations(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    // Set up polling for real-time updates
    const interval = setInterval(loadData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  const handleTrain = async (trainId: string) => {
    try {
      await railwayAPI.handleTrain(trainId);
      Alert.alert('Success', `Train ${trainId} marked as handled`);
      loadData(); // Refresh data
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    }
  };

  const forceRecheck = async (trainId: string) => {
    try {
      await railwayAPI.forceRecheck(trainId);
      Alert.alert('Success', `Force recheck initiated for train ${trainId}`);
      loadData(); // Refresh data
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    }
  };

  const addTrain = async () => {
    try {
      await railwayAPI.registerTrain(newTrain);
      Alert.alert('Success', `Train ${newTrain.train_id} registered successfully`);
      setShowAddTrain(false);
      setNewTrain({
        train_id: '',
        train_type: 'EXPRESS',
        current_section: '',
        from_station: '',
        to_station: '',
        scheduled_arrival: new Date().toISOString(),
        block_length_km: 25.0,
        speed_limit_kmph: 120,
        rake_length_m: 400,
        priority_level: 1,
        headway_seconds: 300
      });
      loadData(); // Refresh data
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    }
  };

  const makeDecision = async () => {
    if (!selectedTrain) return;
    
    try {
      const operatorDecision: OperatorDecision = {
        train_id: selectedTrain.train_id,
        decision: decision.decision,
        delay_minutes: decision.delay_minutes || undefined,
        new_speed_limit: decision.new_speed_limit || undefined,
        reason: decision.reason,
        operator_id: 'OP123', // This should come from login context
        priority: decision.priority
      };

      const response = await railwayAPI.makeOperatorDecision(operatorDecision);
      Alert.alert('Decision Recorded', `Decision for train ${selectedTrain.train_id} has been recorded and driver has been notified.`);
      
      setShowDecisionModal(false);
      setSelectedTrain(null);
      setDecision({
        decision: 'allow_delay',
        delay_minutes: 0,
        new_speed_limit: 120,
        reason: '',
        priority: 'normal'
      });
      loadData(); // Refresh data
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    }
  };

  const openDecisionModal = (train: TrainInfo) => {
    setSelectedTrain(train);
    setShowDecisionModal(true);
  };

  const handleLogout = () => {
    Alert.alert(
      "Logout",
      "Are you sure you want to logout?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Logout", 
          style: "destructive",
          onPress: () => router.replace("/")
        }
      ]
    );
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'CRITICAL': return '#ff4444';
      case 'WATCHLIST': return '#ffaa00';
      case 'SAFE': return '#44ff44';
      default: return '#888888';
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0066cc" />
        <Text style={styles.loadingText}>Loading Railway Data...</Text>
      </View>
    );
  }

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <View>
            <Text style={styles.title}>🚉 Station Operator Dashboard</Text>
            <Text style={styles.subtitle}>Real-time Railway Optimization System</Text>
          </View>
          <TouchableOpacity 
            style={styles.logoutButton}
            onPress={() => {
              console.log("Operator logout button pressed");
              router.replace("/");
            }}
          >
            <Text style={styles.logoutButtonText}>Logout</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* System Status */}
      {health && (
        <View style={styles.statusCard}>
          <Text style={styles.cardTitle}>📡 System Status</Text>
          <View style={styles.statusRow}>
            <Text>Backend: </Text>
            <Text style={[styles.status, { color: health.prediction_engine.running ? '#44ff44' : '#ff4444' }]}>
              {health.prediction_engine.running ? 'ONLINE' : 'OFFLINE'}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text>Models: </Text>
            <Text style={[styles.status, { color: health.models_loaded.delay_model && health.models_loaded.conflict_model ? '#44ff44' : '#ff4444' }]}>
              {health.models_loaded.delay_model && health.models_loaded.conflict_model ? 'LOADED' : 'ERROR'}
            </Text>
          </View>
          <Text style={styles.statusText}>Monitoring {health.prediction_engine.total_trains} trains</Text>
        </View>
      )}

      {/* Critical Situations */}
      {criticalSituations && (
        <View style={styles.criticalCard}>
          <Text style={styles.cardTitle}>🚨 Critical Situations</Text>
          <View style={styles.criticalStats}>
            <View style={styles.statBox}>
              <Text style={styles.statNumber}>{criticalSituations.total_critical}</Text>
              <Text style={styles.statLabel}>Critical</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statNumber}>{criticalSituations.immediate_attention_needed}</Text>
              <Text style={styles.statLabel}>Immediate</Text>
            </View>
          </View>
          
          {criticalSituations.critical_situations.map((situation, index) => (
            <View key={index} style={styles.situationCard}>
              <Text style={styles.trainId}>🚂 {situation.train_id}</Text>
              <Text>Section: {situation.current_section}</Text>
              <Text>Conflict Risk: {safePercentage(situation.conflict_probability)}</Text>
              <Text>Delay: {safeToFixed(situation.delay_prediction, 1, ' min')}</Text>
              {situation.needs_immediate_attention && (
                <Text style={styles.urgentText}>⚠️ IMMEDIATE ATTENTION REQUIRED</Text>
              )}
            </View>
          ))}
        </View>
      )}

      {/* All Trains */}
      <View style={styles.trainsCard}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>🚆 All Trains ({Array.isArray(trains) ? trains.length : 0})</Text>
          <TouchableOpacity 
            style={styles.addButton}
            onPress={() => setShowAddTrain(true)}
          >
            <Text style={styles.addButtonText}>+ Add Train</Text>
          </TouchableOpacity>
        </View>
        
        {Array.isArray(trains) && trains.length > 0 ? (
          trains.map((train, index) => (
          <View key={index} style={styles.trainCard}>
            <View style={styles.trainHeader}>
              <Text style={styles.trainId}>🚂 {train.train_id}</Text>
              <View style={[styles.stateIndicator, { backgroundColor: getStateColor(train.state) }]}>
                <Text style={styles.stateText}>{train.state}</Text>
              </View>
            </View>
            
            <Text>Route: {train.from_station} → {train.to_station}</Text>
            <Text>Section: {train.current_section}</Text>
            <Text>Type: {train.train_type}</Text>
            <Text>Delay: {safeToFixed(train.last_delay_pred, 1, ' min')}</Text>
            <Text>Conflict Risk: {safePercentage(train.last_conflict_prob)}</Text>
            <Text>Last Check: {train.last_check ? new Date(train.last_check).toLocaleTimeString() : 'Never'}</Text>
            
            <View style={styles.actionButtons}>
              <TouchableOpacity 
                style={styles.actionButton}
                onPress={() => handleTrain(train.train_id)}
              >
                <Text style={styles.actionButtonText}>Handle</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.actionButton, styles.recheckButton]}
                onPress={() => forceRecheck(train.train_id)}
              >
                <Text style={styles.actionButtonText}>Recheck</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.actionButton, styles.decisionButton]}
                onPress={() => openDecisionModal(train)}
              >
                <Text style={styles.actionButtonText}>Decide</Text>
              </TouchableOpacity>
            </View>
          </View>
        ))
        ) : (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateText}>📭 No trains registered</Text>
            <Text style={styles.emptyStateSubtext}>Add a train to get started</Text>
          </View>
        )}
      </View>

      {/* Add Train Modal */}
      <Modal visible={showAddTrain} animationType="slide">
        <View style={styles.modalContainer}>
          <Text style={styles.modalTitle}>🚂 Register New Train</Text>
          
          <ScrollView style={styles.modalForm}>
            <TextInput
              style={styles.input}
              placeholder="Train ID (e.g., EXP001)"
              value={newTrain.train_id}
              onChangeText={(text) => setNewTrain({...newTrain, train_id: text})}
            />
            <TextInput
              style={styles.input}
              placeholder="Current Section"
              value={newTrain.current_section}
              onChangeText={(text) => setNewTrain({...newTrain, current_section: text})}
            />
            <TextInput
              style={styles.input}
              placeholder="From Station"
              value={newTrain.from_station}
              onChangeText={(text) => setNewTrain({...newTrain, from_station: text})}
            />
            <TextInput
              style={styles.input}
              placeholder="To Station"
              value={newTrain.to_station}
              onChangeText={(text) => setNewTrain({...newTrain, to_station: text})}
            />
          </ScrollView>
          
          <View style={styles.modalButtons}>
            <TouchableOpacity 
              style={styles.modalButton}
              onPress={() => setShowAddTrain(false)}
            >
              <Text style={styles.modalButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.modalButton, styles.addModalButton]}
              onPress={addTrain}
            >
              <Text style={styles.modalButtonText}>Add Train</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Decision Modal */}
      <Modal visible={showDecisionModal} animationType="slide">
        <View style={styles.modalContainer}>
          <Text style={styles.modalTitle}>⚡ Make Decision for Train {selectedTrain?.train_id}</Text>
          
          <ScrollView style={styles.modalForm}>
            <Text style={styles.inputLabel}>Decision Type</Text>
            <View style={styles.pickerContainer}>
              {['allow_delay', 'reduce_speed', 'hold_train', 'emergency_stop'].map((type) => (
                <TouchableOpacity
                  key={type}
                  style={[
                    styles.pickerOption,
                    decision.decision === type && styles.pickerOptionSelected
                  ]}
                  onPress={() => setDecision({...decision, decision: type})}
                >
                  <Text style={[
                    styles.pickerText,
                    decision.decision === type && styles.pickerTextSelected
                  ]}>
                    {type.replace('_', ' ').toUpperCase()}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {decision.decision === 'allow_delay' && (
              <>
                <Text style={styles.inputLabel}>Delay Minutes</Text>
                <TextInput
                  style={styles.input}
                  placeholder="Enter delay in minutes"
                  value={decision.delay_minutes.toString()}
                  onChangeText={(text) => setDecision({...decision, delay_minutes: parseFloat(text) || 0})}
                  keyboardType="numeric"
                />
              </>
            )}

            {decision.decision === 'reduce_speed' && (
              <>
                <Text style={styles.inputLabel}>New Speed Limit (km/h)</Text>
                <TextInput
                  style={styles.input}
                  placeholder="Enter new speed limit"
                  value={decision.new_speed_limit.toString()}
                  onChangeText={(text) => setDecision({...decision, new_speed_limit: parseFloat(text) || 120})}
                  keyboardType="numeric"
                />
              </>
            )}

            <Text style={styles.inputLabel}>Reason</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Enter reason for this decision"
              value={decision.reason}
              onChangeText={(text) => setDecision({...decision, reason: text})}
              multiline
              numberOfLines={3}
            />

            <Text style={styles.inputLabel}>Priority</Text>
            <View style={styles.pickerContainer}>
              {['normal', 'high', 'critical'].map((priority) => (
                <TouchableOpacity
                  key={priority}
                  style={[
                    styles.pickerOption,
                    decision.priority === priority && styles.pickerOptionSelected
                  ]}
                  onPress={() => setDecision({...decision, priority})}
                >
                  <Text style={[
                    styles.pickerText,
                    decision.priority === priority && styles.pickerTextSelected
                  ]}>
                    {priority.toUpperCase()}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </ScrollView>
          
          <View style={styles.modalButtons}>
            <TouchableOpacity 
              style={styles.modalButton}
              onPress={() => setShowDecisionModal(false)}
            >
              <Text style={styles.modalButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.modalButton, styles.decisionModalButton]}
              onPress={makeDecision}
            >
              <Text style={styles.modalButtonText}>Send Decision</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: '#f5f5f5',
    padding: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  header: {
    backgroundColor: '#0066cc',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  title: { 
    fontSize: 22, 
    fontWeight: "bold", 
    color: 'white',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#e6f2ff',
    textAlign: 'center',
    marginTop: 5,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logoutButton: {
    backgroundColor: '#ff4444',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 5,
  },
  logoutButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  statusCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  status: {
    fontWeight: 'bold',
  },
  statusText: {
    color: '#666',
    fontSize: 12,
    marginTop: 5,
  },
  criticalCard: {
    backgroundColor: '#fff5f5',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
    borderColor: '#ff4444',
    borderWidth: 1,
  },
  criticalStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 10,
  },
  statBox: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ff4444',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
  situationCard: {
    backgroundColor: '#ffe6e6',
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
    borderColor: '#ff9999',
    borderWidth: 1,
  },
  urgentText: {
    color: '#cc0000',
    fontWeight: 'bold',
    fontSize: 12,
    marginTop: 5,
  },
  trainsCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  addButton: {
    backgroundColor: '#28a745',
    padding: 8,
    borderRadius: 6,
  },
  addButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  trainCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    borderColor: '#e9ecef',
    borderWidth: 1,
  },
  trainHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  trainId: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  stateIndicator: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  stateText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  actionButtons: {
    flexDirection: 'row',
    marginTop: 10,
    gap: 10,
  },
  actionButton: {
    backgroundColor: '#007bff',
    padding: 8,
    borderRadius: 6,
    flex: 1,
  },
  recheckButton: {
    backgroundColor: '#6c757d',
  },
  decisionButton: {
    backgroundColor: '#ffc107',
  },
  actionButtonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 12,
    fontWeight: 'bold',
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  pickerContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 15,
    gap: 8,
  },
  pickerOption: {
    backgroundColor: '#e9ecef',
    padding: 10,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#ddd',
    minWidth: 80,
    alignItems: 'center',
  },
  pickerOptionSelected: {
    backgroundColor: '#007bff',
    borderColor: '#007bff',
  },
  pickerText: {
    fontSize: 12,
    color: '#333',
    fontWeight: 'bold',
  },
  pickerTextSelected: {
    color: 'white',
  },
  modalContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: 'white',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  modalForm: {
    flex: 1,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
    fontSize: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 20,
  },
  modalButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    backgroundColor: '#6c757d',
  },
  addModalButton: {
    backgroundColor: '#28a745',
  },
  decisionModalButton: {
    backgroundColor: '#dc3545',
  },
  modalButtonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    marginTop: 20,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
});
