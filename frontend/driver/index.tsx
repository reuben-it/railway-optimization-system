import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from "react-native";
import { DriverNotification, handleAPIError, NotificationsResponse, railwayAPI } from '../services/api';

export default function DriverHome() {
  const [notifications, setNotifications] = useState<DriverNotification[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [driverId] = useState('DRIVER_EXP001'); // This should come from login context

  const loadNotifications = async () => {
    try {
      const response: NotificationsResponse = await railwayAPI.getDriverNotifications(driverId);
      setNotifications(response.notifications);
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadNotifications();
    // Set up polling for real-time updates
    const interval = setInterval(loadNotifications, 15000); // Update every 15 seconds
    return () => clearInterval(interval);
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadNotifications();
  };

  const acknowledgeNotification = async (notificationId: string) => {
    try {
      await railwayAPI.acknowledgeNotification(notificationId);
      Alert.alert('Acknowledged', 'Notification has been acknowledged');
      loadNotifications(); // Refresh data
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    }
  };

  const completeNotification = async (notificationId: string) => {
    try {
      await railwayAPI.completeNotification(notificationId);
      Alert.alert('Completed', 'Action has been marked as completed');
      loadNotifications(); // Refresh data
    } catch (error) {
      Alert.alert('Error', handleAPIError(error));
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return '#ff6b6b';
      case 'acknowledged': return '#ffa726';
      case 'completed': return '#66bb6a';
      default: return '#888888';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return '#d32f2f';
      case 'high': return '#f57c00';
      case 'normal': return '#1976d2';
      default: return '#888888';
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0066cc" />
        <Text style={styles.loadingText}>Loading Notifications...</Text>
      </View>
    );
  }

  const pendingNotifications = notifications.filter(n => n.status === 'pending');
  const acknowledgedNotifications = notifications.filter(n => n.status === 'acknowledged');
  const completedNotifications = notifications.filter(n => n.status === 'completed');

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>🚆 Train Driver Dashboard</Text>
        <Text style={styles.subtitle}>Driver ID: {driverId}</Text>
      </View>

      {/* Summary Cards */}
      <View style={styles.summaryContainer}>
        <View style={[styles.summaryCard, { borderLeftColor: '#ff6b6b' }]}>
          <Text style={styles.summaryNumber}>{pendingNotifications.length}</Text>
          <Text style={styles.summaryLabel}>Pending</Text>
        </View>
        <View style={[styles.summaryCard, { borderLeftColor: '#ffa726' }]}>
          <Text style={styles.summaryNumber}>{acknowledgedNotifications.length}</Text>
          <Text style={styles.summaryLabel}>Acknowledged</Text>
        </View>
        <View style={[styles.summaryCard, { borderLeftColor: '#66bb6a' }]}>
          <Text style={styles.summaryNumber}>{completedNotifications.length}</Text>
          <Text style={styles.summaryLabel}>Completed</Text>
        </View>
      </View>

      {/* Pending Notifications */}
      {pendingNotifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🚨 Urgent - Requires Action</Text>
          {pendingNotifications.map((notification, index) => (
            <View key={index} style={[styles.notificationCard, styles.pendingCard]}>
              <View style={styles.notificationHeader}>
                <Text style={styles.trainId}>🚂 {notification.train_id}</Text>
                <View style={[
                  styles.priorityBadge, 
                  { backgroundColor: getPriorityColor(notification.priority) }
                ]}>
                  <Text style={styles.priorityText}>{notification.priority.toUpperCase()}</Text>
                </View>
              </View>
              
              <Text style={styles.notificationMessage}>{notification.message}</Text>
              <Text style={styles.actionRequired}>Action: {notification.action_required.replace('_', ' ')}</Text>
              <Text style={styles.timestamp}>From: {notification.operator_id} at {notification.timestamp}</Text>
              
              <View style={styles.actionButtons}>
                <TouchableOpacity 
                  style={styles.acknowledgeButton}
                  onPress={() => acknowledgeNotification(notification.notification_id)}
                >
                  <Text style={styles.buttonText}>Acknowledge</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={styles.completeButton}
                  onPress={() => completeNotification(notification.notification_id)}
                >
                  <Text style={styles.buttonText}>Complete</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))}
        </View>
      )}

      {/* Acknowledged Notifications */}
      {acknowledgedNotifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>⏳ Acknowledged - In Progress</Text>
          {acknowledgedNotifications.map((notification, index) => (
            <View key={index} style={[styles.notificationCard, styles.acknowledgedCard]}>
              <View style={styles.notificationHeader}>
                <Text style={styles.trainId}>🚂 {notification.train_id}</Text>
                <View style={[styles.statusBadge, { backgroundColor: getStatusColor(notification.status) }]}>
                  <Text style={styles.statusText}>{notification.status.toUpperCase()}</Text>
                </View>
              </View>
              
              <Text style={styles.notificationMessage}>{notification.message}</Text>
              <Text style={styles.actionRequired}>Action: {notification.action_required.replace('_', ' ')}</Text>
              <Text style={styles.timestamp}>From: {notification.operator_id} at {notification.timestamp}</Text>
              
              <TouchableOpacity 
                style={styles.completeButton}
                onPress={() => completeNotification(notification.notification_id)}
              >
                <Text style={styles.buttonText}>Mark Complete</Text>
              </TouchableOpacity>
            </View>
          ))}
        </View>
      )}

      {/* Completed Notifications */}
      {completedNotifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>✅ Recently Completed</Text>
          {completedNotifications.slice(0, 5).map((notification, index) => (
            <View key={index} style={[styles.notificationCard, styles.completedCard]}>
              <View style={styles.notificationHeader}>
                <Text style={styles.trainId}>🚂 {notification.train_id}</Text>
                <View style={[styles.statusBadge, { backgroundColor: getStatusColor(notification.status) }]}>
                  <Text style={styles.statusText}>{notification.status.toUpperCase()}</Text>
                </View>
              </View>
              
              <Text style={styles.notificationMessage}>{notification.message}</Text>
              <Text style={styles.timestamp}>Completed at {(notification as any).completed_at || notification.timestamp}</Text>
            </View>
          ))}
        </View>
      )}

      {notifications.length === 0 && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>📭 No notifications at this time</Text>
          <Text style={styles.emptyStateSubtext}>All clear! Check back later for updates.</Text>
        </View>
      )}
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
    backgroundColor: '#2e7d32',
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
    color: '#c8e6c9',
    textAlign: 'center',
    marginTop: 5,
  },
  summaryContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  summaryCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 5,
    borderLeftWidth: 4,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  summaryNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  summaryLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
    paddingLeft: 5,
  },
  notificationCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    borderLeftWidth: 4,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  pendingCard: {
    borderLeftColor: '#ff6b6b',
    backgroundColor: '#fff5f5',
  },
  acknowledgedCard: {
    borderLeftColor: '#ffa726',
    backgroundColor: '#fff8e1',
  },
  completedCard: {
    borderLeftColor: '#66bb6a',
    backgroundColor: '#f1f8e9',
  },
  notificationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  trainId: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  priorityText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  notificationMessage: {
    fontSize: 14,
    color: '#333',
    marginBottom: 8,
    lineHeight: 20,
  },
  actionRequired: {
    fontSize: 12,
    color: '#666',
    fontWeight: 'bold',
    marginBottom: 5,
  },
  timestamp: {
    fontSize: 11,
    color: '#999',
    marginBottom: 10,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  acknowledgeButton: {
    backgroundColor: '#ffa726',
    padding: 10,
    borderRadius: 6,
    flex: 1,
  },
  completeButton: {
    backgroundColor: '#66bb6a',
    padding: 10,
    borderRadius: 6,
    flex: 1,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 12,
    fontWeight: 'bold',
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
    backgroundColor: 'white',
    borderRadius: 10,
    marginTop: 20,
  },
  emptyStateText: {
    fontSize: 18,
    color: '#666',
    marginBottom: 10,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
});
