import { StyleSheet, Text, View } from "react-native";


export default function DriverHome() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>🚆 Train Driver Dashboard</Text>
      <Text>Here you can view your assigned train schedule and instructions.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", alignItems: "center" },
  title: { fontSize: 22, fontWeight: "bold", marginBottom: 10 },
});
