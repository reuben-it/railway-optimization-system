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
      <Text style={styles.title}>🔐 User Sign In</Text>

      <TextInput
        placeholder="Login ID"
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

      <TouchableOpacity style={styles.button} onPress={() => router.push("/station1")} />
        <Text style={styles.buttonText}>"Station1"</Text>

      <TouchableOpacity style={styles.button} onPress={handleSignIn}>
        <Text style={styles.buttonText}>Sign In</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", alignItems: "center", padding: 20 },
  title: { fontSize: 24, fontWeight: "bold", marginBottom: 30 },
  input: {
    width: "100%",
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 6,
    padding: 12,
    marginBottom: 15,
    backgroundColor: "white",
  },
  button: {
    width: "100%",
    backgroundColor: "#2563eb",
    padding: 12,
    borderRadius: 6,
    alignItems: "center",
  },
  buttonText: { color: "white", fontWeight: "bold", fontSize: 16 },
});
