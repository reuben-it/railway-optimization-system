import { useRouter } from "expo-router";
import { useState } from "react";
import {
  Alert,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

export default function SignIn() {
  const [loginId, setLoginId] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  // Example mock user database
  const users: Record<string, { password: string; role: "operator" | "driver" }> = {
    "OP123": { password: "op@123", role: "operator" },
    "OP456": { password: "station@456", role: "operator" },
    "DR123": { password: "dr@123", role: "driver" },
    "DR456": { password: "train@456", role: "driver" },
  };

  const handleSignIn = () => {
    const user = users[loginId];
    if (!user || user.password !== password) {
      Alert.alert("Error", "Invalid Login ID or password.");
      return;
    }

    // Redirect based on role
    if (user.role === "operator") {
      router.replace("/operator");
    } else if (user.role === "driver") {
      router.replace("/driver");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🚂 Railway Optimization System</Text>
      <Text style={styles.subtitle}>Role-Based Access Control</Text>

      <TextInput
        placeholder="Login ID (OP123, DR123, etc.)"
        value={loginId}
        onChangeText={setLoginId}
        style={styles.input}
        autoCapitalize="characters"
      />

      <TextInput
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        style={styles.input}
        secureTextEntry
      />

      <TouchableOpacity style={styles.button} onPress={handleSignIn}>
        <Text style={styles.buttonText}>Sign In</Text>
      </TouchableOpacity>

      <View style={styles.helpContainer}>
        <Text style={styles.helpTitle}>Demo Accounts:</Text>
        <Text style={styles.helpText}>Operator: OP123 / op@123</Text>
        <Text style={styles.helpText}>Driver: DR123 / dr@123</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    justifyContent: "center", 
    alignItems: "center", 
    padding: 20,
    backgroundColor: '#f0f8ff'
  },
  title: { 
    fontSize: 28, 
    fontWeight: "bold", 
    marginBottom: 10,
    color: '#1e40af',
    textAlign: 'center'
  },
  subtitle: {
    fontSize: 16,
    color: '#64748b',
    marginBottom: 30,
    textAlign: 'center'
  },
  input: {
    width: "100%",
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
    backgroundColor: "white",
    fontSize: 16,
  },
  button: {
    width: "100%",
    backgroundColor: "#1e40af",
    padding: 15,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 10,
  },
  buttonText: { 
    color: "white", 
    fontWeight: "bold", 
    fontSize: 16 
  },
  helpContainer: {
    marginTop: 30,
    padding: 20,
    backgroundColor: 'white',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  helpTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#374151',
  },
  helpText: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 5,
  },
});
